import datetime
from base64 import b64encode
import os
from re import sub as resub

from dagster import (
    asset, get_dagster_logger, job, op,
    DynamicOut, DynamicOutput,
)
import iris
import iris.coord_categorisation as iccat
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
    radar_agg_3hr = radar_cube.copy()
    iccat.add_hour(radar_agg_3hr, coord='time')
    iccat.add_day_of_year(radar_agg_3hr, coord='time')
    iccat.add_categorised_coord(radar_agg_3hr, "3hr", "hour", lambda _, value: value // 3)
    radar_agg_3hr.data = radar_agg_3hr.data * (1.0 / 12.0)
    return radar_agg_3hr


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
    radar_cube = radar_cube(datasets.collect())