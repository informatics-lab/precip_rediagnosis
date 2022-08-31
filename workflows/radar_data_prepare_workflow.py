from base64 import b64encode
import datetime
import functools
import os
from re import sub as resub

from dagster import (
    asset, get_dagster_logger, job, make_values_resource, op, graph,
    DynamicOut, DynamicOutput,
    Out,
)
import iris
import iris.coord_categorisation as iccat
from iris.cube import Cube, CubeList
import numpy as np
import pandas as pd
import xarray as xr


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


##########
#
# Loading of data assets.
#
##########


@op(required_resource_keys={"setup"})
def load_input_dataset(context, date):
    """Load a radar data file for a single time point."""
    filepath = context.resources.setup["data_path"]
    filename = context.op_config["dataset_filename_template"].format(
        product=context.op_config["product"],
        dt=date
    )
    return iris.load_cube(os.path.join(filepath, filename))


@op
def iris_concatenate(cubes):
    """Concatenate a list of Iris cubes to a single resultant cube."""
    return CubeList(cubes).concatenate_cube()



@asset
def radar_cube(cubes):
    """Concatenated radar data cube - the workflow's data input."""
    return iris_concatenate(cubes)


@asset
def radar_cube_latlon(lat_vals, lon_vals, radar_cube):
    """Radar cube with latitude and longitude coords added."""
    return add_latlon_coords(lat_vals, lon_vals, radar_cube)


@asset
def radar_cube_3hr(radar_cube):
    """Aggregate radar data into three-hour windows."""
    iccat.add_hour(radar_cube, coord='time')
    iccat.add_day_of_year(radar_cube, coord='time')
    iccat.add_categorised_coord(radar_cube, "3hr", "hour", lambda _, value: value // 3)
    radar_agg_3hr = radar_cube.aggregated_by(['3hr', 'day_of_year'], iris.analysis.SUM)
    radar_agg_3hr.data = radar_agg_3hr.data * (1.0 / 12.0)
    for coord_name in ['forecast_reference_time', 'hour', 'day_of_year', '3hr']:
        radar_agg_3hr.remove_coord(coord_name)
    return radar_agg_3hr


@op(required_resource_keys={"setup"})
def locate_target_grid_cube(context):
    """Inject filepath and standard name of grid cube to load into the workflow."""
    filepath = context.resources.setup["data_path"]
    filename = context.op_config["filename"]
    return os.path.join(filepath, filename)


@asset
def target_grid_cube(full_filepath):
    """Define the grid cube as a workflow asset."""
    return iris.load_cube(full_filepath)


##########
#
# Metadata handling.
#
##########


@op(out={"lat_vals": Out(), "lon_vals": Out()})
def calc_lat_lon_coords(radar_cube, target_grid_cube):
    """Define latitude and longitude coord values for the radar cube."""
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
    """Add auxiliary latitude and longitude coords to the radar cube."""
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


# @op(out={"lat_target_index": Out(), "lon_target_index": Out(), "num_cells": Out()})
# def calc_target_cube_indices(lat_vals, lon_vals, radar_cube, target_grid_cube):
#     """
#     Calculate the latitude and longitude index in the target cube
#     coordinate system of each grid square in the radar cube.
#     :param lat_vals: A 1D array of the target latitude values
#     :param lon_vals: A 1D array of the target longitude values
#     :param radar_cube: The source radar cube for the calculating the mapping
#     :return: 2D np arrays with a mapping for each cell in the radar
#     cube to the index in latitude and longitude of the target cube.
#     """
#     lat_target_index = -1 * np.ones(
#         (radar_cube.shape[1], radar_cube.shape[2]),
#         dtype='int32',
#     )
#     lon_target_index = -1 * np.ones(
#         (radar_cube.shape[1], radar_cube.shape[2]),
#         dtype='int32',
#     )

#     num_cells = np.zeros((target_grid_cube.shape[0], target_grid_cube.shape[1]))
#     for i_lon, bnd_lon in enumerate(target_grid_cube.coord('longitude').bounds):
#         for i_lat, bnd_lat in enumerate(target_grid_cube.coord('latitude').bounds):
#             arr1, arr2 = np.where(
#                 (lat_vals >= bnd_lat[0]) &
#                 (lat_vals < bnd_lat[1]) &
#                 (lon_vals >= bnd_lon[0]) &
#                 (lon_vals < bnd_lon[1])
#             )
#             lon_target_index[arr1, arr2] = i_lon
#             lat_target_index[arr1, arr2] = i_lat
#             num_cells[i_lat, i_lon] = len(arr1)
#     return lat_target_index, lon_target_index, num_cells


@op(out={"lat_target_index": Out(), "lon_target_index": Out()})
def calc_target_cube_indices(lat_vals, lon_vals, radar_cube, target_grid_cube):
    """
    Calculate the latitude and longitude index in the target cube
    coordinate system of each grid square in the radar cube.
    :param lat_vals: A 1D array of the target latitude values
    :param lon_vals: A 1D array of the target longitude values
    :param radar_cube: The source radar cube for the calculating the mapping
    :return: 2D np arrays with a mapping for each cell in the radar
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

    # num_cells = np.zeros((target_grid_cube.shape[0], target_grid_cube.shape[1]))
    for i_lon, bnd_lon in enumerate(target_grid_cube.coord('longitude').bounds):
        for i_lat, bnd_lat in enumerate(target_grid_cube.coord('latitude').bounds):
            arr1, arr2 = np.where(
                (lat_vals >= bnd_lat[0]) & (lat_vals < bnd_lat[1]) &
                (lon_vals >= bnd_lon[0]) & (lon_vals < bnd_lon[1])
            )
            lon_target_index[arr1, arr2] = i_lon
            lat_target_index[arr1, arr2] = i_lat
            # num_cells[i_lat, i_lon] = len(arr1)
    return lat_target_index, lon_target_index


##########
#
# Regrid prep.
#
##########


@op(required_resource_keys={"regrid"})
def get_var_names(context):
    """Inject var names from config into the workflow."""
    return context.resources.regrid["var_names"]


# def regrid_array(context, key):
@op(required_resource_keys={"setup", "regrid"})
def regrid_array(context, key, dates, tgt_grid_cube):
    """Construct a regrid array appropriate for the array type."""
    rainfall_thresholds = context.resources.setup["rainfall_thresholds"]
    var_names = context.resources.regrid["var_names"]
    var_types = context.resources.regrid["var_types"]

    n_times = len(dates)
    tgt_lat_shape, = tgt_grid_cube.coord("latitude").shape
    tgt_lon_shape, = tgt_grid_cube.coord("longitude").shape
    # Index to find the appropriate variable type for the received variable name key.
    array_type = var_types[var_names.index(key)]
    if array_type == "VECTOR":
        a = np.zeros([n_times, tgt_lat_shape, tgt_lon_shape, len(rainfall_thresholds)])
    elif array_type == "MASK_VECTOR":
        a = np.ones([n_times, tgt_lat_shape, tgt_lon_shape, len(rainfall_thresholds)])
    elif array_type == "SCALAR":
        a = np.zeros([n_times, tgt_lat_shape, tgt_lon_shape])
    elif array_type == "MASK_SCALAR":
        a = np.ones([n_times, tgt_lat_shape, tgt_lon_shape])
    else:
        raise ValueError(f"Bad array type {array_type!r}")
    return a


@op(required_resource_keys={"regrid"})
def gather_regrid_arrays(context, arrays):
    """Combine each individual array of regridded data into a single dict."""
    var_names = context.resources.regrid["var_names"]
    result = {n: a for n, a in zip(var_names, arrays)}
    get_dagster_logger().info(f"Regrid arrays: {result}")
    return result


##########
#
# Regrid operations.
#
##########


@op(required_resource_keys={"setup"})
def regrid(
    context,
    radar_cube, radar_agg_3hr, tgt_grid_cube,
    validity_times, regridded_arrays_dict,
    lon_target_index, lat_target_index
):
    """Regrid the radar data onto the MOGREPs-G grid."""
    rainfall_thresholds = context.resources.setup["rainfall_thresholds"]
    def compare_time(t1, t2):
        return (t1.year==t2.year) and (t1.month==t2.month) and (t1.day==t2.day) and (t1.hour==t2.hour) and (t1.minute==t2.minute)

    get_dagster_logger().debug('Performing regrid from radar to target grid.')
    # Iterate through each time, rain amount band, latitude and longitude
    for i_time, validity_time in enumerate(validity_times):
        get_dagster_logger().info(f'Processing radar data for validity time {validity_time}')
        radar_select_time = radar_agg_3hr.extract(iris.Constraint(
            time=lambda c1: compare_time(c1.bound[0], validity_time)
        ))
        masked_radar = np.ma.MaskedArray(radar_select_time.data.data, radar_agg_3hr[0].data.mask)

        radar_instant_select_time = radar_cube.extract(iris.Constraint(
            time=lambda c1: compare_time(c1.point, validity_time)
        ))
        masked_radar_instant = np.ma.MaskedArray(
            radar_instant_select_time.data.data,
            radar_cube[0].data.mask
        )
        for i_lat in range(tgt_grid_cube.shape[0]):
            for i_lon in range(tgt_grid_cube.shape[1]):
                selected_cells = (
                    (~(radar_select_time.data.mask)) &
                    (lat_target_index == i_lat) &
                    (lon_target_index == i_lon)
                )
                masked_radar.mask = ~selected_cells
                masked_radar_instant.mask = ~selected_cells

                radar_cells_in_mg = np.count_nonzero(selected_cells)
                # Only proceed with processing for this tagret grid cell
                # if there are some radar grid cells within this target
                # grid cell.
                # get_dagster_logger().debug(f'{i_lat}, {i_lon}')
                if radar_cells_in_mg > 0:
                    # set the values for this location to be unmasker,
                    # as we have valid radar values for this location
                    regridded_arrays_dict['bands_mask'][i_time, i_lat, i_lon, :] = False
                    regridded_arrays_dict['scalar_value_mask'][i_time, i_lat, i_lon] = False
                    for imp_ix, (_, imp_bounds) in enumerate(rainfall_thresholds.items()):
                        # calculate fraction in band for 3 hour aggregate data
                        num_in_band_agg = np.count_nonzero(
                            (masked_radar.compressed() >= imp_bounds[0]) &
                            (masked_radar.compressed() <= imp_bounds[1])
                        )
                        regridded_arrays_dict['radar_fraction_in_band_aggregate_3hr'][
                            i_time, i_lat, i_lon, imp_ix] = num_in_band_agg / (len(masked_radar.compressed()))

                        # calculate fraction in band for instant radar data
                        num_in_band_instant = np.count_nonzero(
                            (masked_radar_instant.compressed() >= imp_bounds[0]) &
                            (masked_radar_instant.compressed() <= imp_bounds[1])
                        )
                        regridded_arrays_dict['radar_fraction_in_band_instant'][i_time, i_lat, i_lon, imp_ix] = \
                            num_in_band_instant / (len(masked_radar_instant.compressed()))

                    regridded_arrays_dict['fraction_sum_agg'][i_time, i_lat, i_lon] = \
                        regridded_arrays_dict['radar_fraction_in_band_aggregate_3hr'][i_time, i_lat, i_lon, :].sum()
                    # get_dagster_logger().debug(f'sum of fractions agg {regridded_arrays_dict["fraction_sum_agg"][i_time, i_lat, i_lon] }')
                    regridded_arrays_dict['fraction_sum_instant'][i_time, i_lat, i_lon] = \
                        regridded_arrays_dict['radar_fraction_in_band_instant'][i_time, i_lat, i_lon, :].sum()
                    # get_dagster_logger().debug(f'sum of fractions instant {regridded_arrays_dict["fraction_sum_instant"][i_time, i_lat, i_lon] }')

                    # calculate the max and average of all radar cells within each mogreps-g cell
                    regridded_arrays_dict['radar_max_rain_aggregate_3hr'][i_time, i_lat, i_lon] = masked_radar.max()
                    regridded_arrays_dict['radar_mean_rain_aggregate_3hr'][i_time, i_lat, i_lon] = \
                        masked_radar.sum() / radar_cells_in_mg
                    # get_dagster_logger().debug(f'{regridded_arrays_dict["radar_mean_rain_aggregate_3hr"][i_time, i_lat, i_lon]} , {regridded_arrays_dict["radar_max_rain_aggregate_3hr"][i_time, i_lat, i_lon]},' )

                    # create instant radar rate feature data
                    regridded_arrays_dict['radar_max_rain_instant'][i_time, i_lat, i_lon] = masked_radar_instant.max()
                    regridded_arrays_dict['radar_mean_rain_instant'][i_time, i_lat, i_lon] = \
                        masked_radar_instant.sum() / radar_cells_in_mg
                    # get_dagster_logger().debug(f'{regridded_arrays_dict["radar_mean_rain_instant"][i_time, i_lat, i_lon]} , {regridded_arrays_dict["radar_max_rain_instant"][i_time, i_lat, i_lon]},')
                else:
                    get_dagster_logger().info(f'No radar cells to include at ({i_lat}, {i_lon}).')

    total_num_pts = (
        regridded_arrays_dict['fraction_sum_instant'].shape[0] *
        regridded_arrays_dict['fraction_sum_instant'].shape[1] *
        regridded_arrays_dict['fraction_sum_instant'].shape[2]
    )
    get_dagster_logger().info(
        f'Sum of fraction aggregate, number equal to 1: '
        f'{(regridded_arrays_dict["fraction_sum_agg"] > 0.999).sum()} of {total_num_pts}'
    )
    get_dagster_logger().info(
        f'Sum of fraction instant, number equal to 1: '
        f'{(regridded_arrays_dict["fraction_sum_instant"] > 0.999).sum()} of {total_num_pts}'
    )
    return regridded_arrays_dict


##########
#
# Regrid post-processing to produce CSV result.
#
##########

@op(
    required_resource_keys={"setup"},
    out={"band_coord": Out(), "radar_time_coord": Out()}
)
def build_extra_coords(context, validity_times, radar_cube):
    """Define extra coordinates for the cubes of regridded data."""
    rainfall_thresholds = context.resources.setup["rainfall_thresholds"]
    band_coord = iris.coords.AuxCoord(
        [float(t) for t in rainfall_thresholds.keys()],
        bounds=list(rainfall_thresholds.values()),
        var_name='band',
        units='mm',
    )
    radar_time_coord = iris.coords.DimCoord(
        [vt.timestamp() for vt in validity_times],
        var_name='time',
        units=radar_cube.coord('time').units,
    )
    return band_coord, radar_time_coord


# @op
# def build_num_cells_cube(num_cells, target_lat_coord, target_lon_coord):
#     # XXX output not used!
#     num_cells_cube = iris.cube.Cube(
#             data=num_cells,
#             dim_coords_and_dims=(
#              (target_lat_coord, 0), (target_lon_coord, 1),),
#             var_name='num_radar_cells',
#         )
#     return num_cells_cube


@op(out={"coords": Out()})
def collate_coords(target_grid_cube, radar_time_coord, band_coord):
    """
    Make a list of the coords needed for the regridded data cubes
    for ease of passing to downstream functions.

    """
    lat_coord = target_grid_cube.coord('latitude')
    lon_coord = target_grid_cube.coord('longitude')
    return (radar_time_coord, lat_coord, lon_coord, band_coord)


@op(
    required_resource_keys={"regrid"},
    out={"vector_var_names": Out(), "scalar_var_names": Out()}
)
def get_arrays_by_type(context):
    """Sort array names by whether they describe scalar or vector data."""
    var_names = context.resources.regrid["var_names"]
    var_types = context.resources.regrid["var_types"]
    target_var_types = ["VECTOR", "SCALAR"]
    names = []
    for array_type in target_var_types:
        names.append([n for (n, t) in zip(var_names, var_types) if t == array_type])
    return tuple(names[0]), tuple(names[1])


@op(required_resource_keys={"regrid"})
def build_vector_cubes(context, var_name, regridded_arrays_dict, coords):
    """Construct Iris cubes of vector regridded data arrays."""
    var_names = context.resources.regrid["var_names"]
    long_names = context.resources.regrid["output_long_names"]
    long_name = long_names[var_names.index(var_name)]

    data = np.ma.MaskedArray(
        data=regridded_arrays_dict[var_name],
        mask=regridded_arrays_dict['bands_mask']
    )
    radar_time_coord, target_lat_coord, target_lon_coord, band_coord = coords
    dcad = (
        (radar_time_coord, 0),
        (target_lat_coord, 1),
        (target_lon_coord, 2),
    )
    acad = [(band_coord, 3)]
    return Cube(
        data=data,
        dim_coords_and_dims=dcad,
        aux_coords_and_dims=acad,
        units=None,
        var_name=var_name,
        long_name=long_name
    )


@op(required_resource_keys={"regrid"})
def build_scalar_cubes(context, var_name, regridded_arrays_dict, coords):
    """Construct Iris cubes of scalar regridded data arrays."""
    var_names = context.resources.regrid["var_names"]
    long_names = context.resources.regrid["output_long_names"]
    long_name = long_names[var_names.index(var_name)]

    data = np.ma.MaskedArray(
        data=regridded_arrays_dict[var_name],
        mask=regridded_arrays_dict['scalar_value_mask'],
    )
    radar_time_coord, target_lat_coord, target_lon_coord, _ = coords
    # cl = {c.name(): len(c.points) for c in coords[:-1]}
    dcad = (
        (radar_time_coord, 0),
        (target_lat_coord, 1),
        (target_lon_coord, 2)
    )
    return Cube(
        data=data,
        dim_coords_and_dims=dcad,
        units='mm',
        var_name=var_name,
        long_name=long_name
    )


@op
def build_regridded_cubelist(vector_cubes, scalar_cubes):
    """Build a `CubeList` of the vector and scalar datasets."""
    scalar_cubes.extend(vector_cubes)
    return CubeList(scalar_cubes)


##########
#
# Convert post-processed data to CSV.
#
##########


@op
def cube_to_dataframe(var_name, cubes):
    """Convert an Iris cube to a pandas DataFrame using XArray."""
    cube, = cubes.extract(var_name)
    return xr.DataArray.from_iris(cube).to_dataframe().reset_index()


@op
def process_scalar_df(scalar_dfs):
    """Produce the dataframe of scalar data values."""
    return functools.reduce(
        lambda x, y: pd.merge(x, y, on=('latitude', 'longitude', 'time')),
        scalar_dfs
    )


@op(required_resource_keys={"setup", "regrid"})
def process_vector_df(context, data_df, vector_dfs):
    """Combine scalar and vector dataframes."""
    rain_bands = context.resources.setup["rainfall_thresholds"]
    var_names = context.resources.regrid["var_names"]
    var_types = context.resources.regrid["var_types"]
    vector_var_names = [n for (n, t) in zip(var_names, var_types) if t == "VECTOR"]
    for i, vector_df in enumerate(vector_dfs):
        var_name = vector_var_names[i]  # Rely on consistent ordering of vector vars.
        for band in rain_bands.keys():
            df = vector_df[vector_df['band'] == float(band)][['time', 'latitude', 'longitude', var_name]]
            df = df.rename({var_name: f'{var_name}_{band}'}, axis='columns')
            data_df = pd.merge(data_df, df, on=['time', 'latitude', 'longitude'])
    return data_df


@op(required_resource_keys={"regrid"})
def df_nan_exclude(context, data_df):
    """Exclude NaN values from the resultant dataframe."""
    var_names = context.resources.regrid["var_names"]
    var_types = context.resources.regrid["var_types"]
    scalar_selected_var = [n for (n, t) in zip(var_names, var_types) if t == "SCALAR"][0]
    return data_df[~(data_df[scalar_selected_var].isna())]


##########
#
# Store cube and CSV results.
#
##########


@op
def filename_timestamp(context, dates):
    """Format the timestamp to add to the saved artifacts' filenames."""
    filename_template = context.op_config["date_fname_template"]
    return filename_template.format(start=dates[0], end=dates[-1])


@op(required_resource_keys={"setup"})
def iris_save(context, cubelist, filename):
    """Save the regridded data to NetCDF using Iris."""
    radar_data_dir = context.resources.setup["data_path"]
    filename_prefix = context.resources.setup["radar_fname_prefix"]
    full_filename = f"{filename_prefix}_{filename}.nc"
    iris.save(cubelist, os.path.join(radar_data_dir, full_filename))


@op(required_resource_keys={"setup"})
def save_csv(context, data_df, filename):
    """Save the flattened dataframe of regridded data to CSV."""
    radar_data_dir = context.resources.setup["data_path"]
    filename_prefix = context.resources.setup["radar_fname_prefix"]
    radar_csv_filename = f"{filename_prefix}_{filename}.csv"
    data_df.to_csv(os.path.join(radar_data_dir, radar_csv_filename), index=False)


##########
#
# Job definition and helper functions.
#
##########


# XXX: should be shared with extract workflow, not duplicated.
@op(out=DynamicOut())
def dynamicise(l):
    """Convert an ordinary Python iterable `l` into a dagster dynamic output."""
    # Make a unique keyname from ascii-only elements of random strings from `/dev/urandom`.
    key = resub(r"\W", "", b64encode(os.urandom(16)).decode("utf-8")[:16])
    for i, l_itm in enumerate(l):
        yield DynamicOutput(l_itm, mapping_key=f"{key}_{i}")


@job(
    resource_defs={
        "setup": make_values_resource(
            data_path=str,
            radar_fname_prefix=str,
            rainfall_thresholds=dict,
        ),
        "regrid": make_values_resource(
            var_names=list,
            var_types=list,
            output_long_names=list,
        )
    }
)
def radar_preprocess():
    """Define the workflow used to process radar data."""
    # Load relevant data (within the required time window).
    dates = dates_to_extract()
    datasets = dynamicise(dates).map(load_input_dataset)
    rcube = radar_cube(datasets.collect())
    tgt_grid_cube = target_grid_cube(locate_target_grid_cube())

    # Add required extra metadata.
    lat_vals, lon_vals = calc_lat_lon_coords(rcube, tgt_grid_cube)
    rcube_latlon = radar_cube_latlon(lat_vals, lon_vals, rcube)
    rcube_agg_3hr = radar_cube_3hr(rcube_latlon)

    # Regrid prep.
    lat_target_index, lon_target_index = calc_target_cube_indices(
        lat_vals, lon_vals, rcube_latlon, tgt_grid_cube
    )
    var_names = dynamicise(get_var_names())
    arrays = var_names.map(lambda k: regrid_array(k, dates, tgt_grid_cube))
    regrid_arrays_dict = gather_regrid_arrays(arrays.collect())

    # Regrid.
    regrid_data_dict = regrid(
        rcube_latlon, rcube_agg_3hr, tgt_grid_cube,
        dates, regrid_arrays_dict,
        lon_target_index, lat_target_index
    )

    # Regrid post-processing.
    band_coord, radar_time_coord = build_extra_coords(dates, rcube_latlon)
    # XXX Output not used elsewhere in workflow? Can remove `num_cells` too if not used.
    # num_cells_cube = build_num_cells_cube(num_cells, lat_target_index, lon_target_index)
    regrid_coords = collate_coords(tgt_grid_cube, radar_time_coord, band_coord)
    vector_var_names, scalar_var_names = get_arrays_by_type()
    vector_names = dynamicise(vector_var_names)
    scalar_names = dynamicise(scalar_var_names)
    vector_cubes = vector_names.map(lambda n: build_vector_cubes(n, regrid_data_dict, regrid_coords))
    scalar_cubes = scalar_names.map(lambda n: build_scalar_cubes(n, regrid_data_dict, regrid_coords))
    regridded_cubes = build_regridded_cubelist(vector_cubes.collect(), scalar_cubes.collect())

    # Convert post-processed data to CSV.
    vector_dfs = vector_names.map(lambda n: cube_to_dataframe(n, regridded_cubes)).collect()
    scalar_dfs = scalar_names.map(lambda n: cube_to_dataframe(n, regridded_cubes)).collect()
    # Avoid re-defining `radar_df` with each op call.
    radar_df = df_nan_exclude(process_vector_df(process_scalar_df(scalar_dfs), vector_dfs))

    # Store results.
    filename = filename_timestamp(dates)
    iris_save(regridded_cubes, filename)
    save_csv(radar_df, filename)