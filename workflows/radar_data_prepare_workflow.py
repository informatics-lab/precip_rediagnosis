import datetime
from base64 import b64encode
import os
from re import sub as resub

from dagster import (
    asset, get_dagster_logger, job, op,
    DynamicOut, DynamicOutput,
    Out,
)
import iris
import iris.coord_categorisation as iccat
import numpy as np
import pandas as pd


# XXX: should be shared with extract workflow, not duplicated.
@op
def dates_to_extract(context):
    """Generate a list of dates to define files to extract from MASS."""
    datetime_str = context.op_config["datetime_str"]
    delta_hours = context.op_config["archive_time_chunk"]
    event_start = datetime.datetime.strptime(
        context.op_config["event_start"], datetime_str
    )
    event_end = datetime.datetime.strptime(
        context.op_config["event_end"], datetime_str
    )
    start_date = datetime.datetime(
        event_start.year,
        event_start.month,
        event_start.day,
        0, 0
    )
    end_date = datetime.datetime(
        event_end.year,
        event_end.month,
        event_end.day,
        23, 59
    )
    dates = list(pd.date_range(
        start=start_date,
        end=end_date,
        freq=datetime.timedelta(hours=delta_hours)
    ).to_pydatetime())
    return dates


@op
def load_input_dataset(context, date):
    filepath = context.op_config["data_path"]
    filename = context.op_config["dataset_filename_template"].format(
        product=context.op_config["product"],
        dt=date
    )
    return iris.load_cube(os.path.join(filepath, filename))


@op
def iris_concatenate(cubes):
    return iris.cube.CubeList(cubes).concatenate_cube()



@asset
def radar_cube(cubes):
    return iris_concatenate(cubes)


@asset
def radar_cube_3hr(radar_cube):
    # radar_agg_3hr = radar_cube.copy()
    iccat.add_hour(radar_cube, coord='time')
    iccat.add_day_of_year(radar_cube, coord='time')
    iccat.add_categorised_coord(radar_cube, "3hr", "hour", lambda _, value: value // 3)
    radar_agg_3hr = radar_cube.aggregated_by(['3hr', 'day_of_year'], iris.analysis.SUM)
    radar_agg_3hr.data = radar_agg_3hr.data * (1.0 / 12.0)  # Very memory-intensive!
    for coord_name in ['forecast_reference_time', 'hour', 'day_of_year', '3hr']:
        radar_agg_3hr.remove_coord(coord_name)
    return radar_agg_3hr


@op
def locate_target_grid_cube(context):
    filepath = context.op_config["data_path"]
    filename = context.op_config["dataset_filename_template"]
    return os.path.join(filepath, filename)


@asset
def target_grid_cube(full_filepath):
    return iris.load_cube(full_filepath)


@op(out={"lat_vals": Out(), "lon_vals": Out()})
def calc_lat_lon_coords(radar_cube, target_grid_cube):
    get_dagster_logger().info('Calculating index mapping for target grid')
    radar_crs = radar_cube.coord_system().as_cartopy_crs()
    # Create some helper arrays for converting from our radar grid to the mogreps-g grid
    X_radar, Y_radar = np.meshgrid(
        radar_cube.coord('projection_x_coordinate').points,
        radar_cube.coord('projection_y_coordinate').points
    )
    target_crs = target_grid_cube.coord_system().as_cartopy_crs()
    ret_val = target_crs.transform_points(
        radar_crs,
        X_radar,
        Y_radar
    )
    lat_vals = ret_val[:, :, 1]
    lon_vals = ret_val[:, :, 0]
    return lat_vals, lon_vals


@op
def add_latlon_coords(lat_vals, lon_vals, cube):
    lon_coord = iris.coords.AuxCoord(
        lon_vals,
        standard_name='longitude',
        units='degrees',
    )
    lat_coord = iris.coords.AuxCoord(
        lat_vals,
        standard_name='latitude',
        units='degrees',
    )
    cube.add_aux_coord(lon_coord, [1, 2])
    cube.add_aux_coord(lat_coord, [1, 2])
    return cube


@op(out={"lat_target_index": Out(), "lon_target_index": Out(), "num_cells": Out()})
def calc_target_cube_indices(lat_vals, lon_vals, radar_cube, target_grid_cube):
    """
    Calculate the latitude and longitude index in the target cube
    coordinate system of each grid square in the radar cube.
    :param lat_vals: A 1D array of the target latitude values
    :param lon_vals: A 1D array of the target longitude values
    :param radar_cube: The source radar cube for the calculating the mapping
    :return: 2D numpy arrays with a mapping for each cell in the radar
    cube to the index in latitude and longitude of the target cube.
    """
    lat_target_index = -1 * np.ones(
        (radar_cube.shape[1], radar_cube.shape[2]),
        dtype='int32',
    )
    lon_target_index = -1 * np.ones(
        (radar_cube.shape[1], radar_cube.shape[2]),
        dtype='int32',
    )

    num_cells = np.zeros((target_grid_cube.shape[0], target_grid_cube.shape[1]))
    for i_lon, bnd_lon in enumerate(target_grid_cube.coord('longitude').bounds):
        for i_lat, bnd_lat in enumerate(target_grid_cube.coord('latitude').bounds):
            arr1, arr2 = np.where(
                (lat_vals >= bnd_lat[0]) &
                (lat_vals < bnd_lat[1]) &
                (lon_vals >= bnd_lon[0]) &
                (lon_vals < bnd_lon[1])
            )
            lon_target_index[arr1, arr2] = i_lon
            lat_target_index[arr1, arr2] = i_lat
            num_cells[i_lat, i_lon] = len(arr1)
    return lat_target_index, lon_target_index, num_cells


# XXX: should be shared with extract workflow, not duplicated.
@op(out=DynamicOut())
def dynamicise(l):
    """Convert an ordinary Python iterable `l` into a dagster dynamic output."""
    key = resub(r"\W", "", b64encode(os.urandom(16)).decode("utf-8")[:16])
    for i, l_itm in enumerate(l):
        yield DynamicOutput(l_itm, mapping_key=f"{key}_{i}")


@job
def radar_preprocess():
    dates = dynamicise(dates_to_extract())
    datasets = dates.map(load_input_dataset)
    rcube = radar_cube(datasets.collect())
    rcube_agg_3hr = radar_cube_3hr(rcube)
    tgt_grid_cube = target_grid_cube(locate_target_grid_cube())

    lat_vals, lon_vals = calc_lat_lon_coords(rcube, tgt_grid_cube)
    lat_target_index, lon_target_index, num_cells = calc_target_cube_indices(
        lat_vals, lon_vals, radar_cube
    )
    cubes = dynamicise([rcube, rcube_agg_3hr])
    rcube, rcube_agg_3hr = cubes.map(lambda c: add_latlon_coords(lat_vals, lon_vals, c)).collect()