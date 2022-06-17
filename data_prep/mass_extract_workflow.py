import datetime
import os
import subprocess

from dagster import (
    get_dagster_logger,
    graph, job, op,
    DynamicOut, DynamicOutput,
    In, Out,
    RetryRequested
)
import pandas as pd


@op
def dates_to_extract(context):
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


@op(out=DynamicOut())
def get_mass_paths(context, dates):
    mass_root = context.op_config["mass_root"]
    mass_filename_template = context.op_config["mass_filename_template"]
    for i, dt in enumerate(dates):
        mass_path = os.path.join(mass_root, mass_filename_template.format(dt=dt))
        yield DynamicOutput(mass_path, mapping_key=i)


@op
def retrieve_from_mass(context, mass_path, mass_tar_name):
    mass_get_cmd_template = context.op_config["mass_get_cmd"]
    mass_get_args = context.op_config["mass_get_args"]
    dest_path = os.path.join(context.op_config["dest_path"], mass_tar_name)
    mass_get_cmd = mass_get_cmd_template.format(
        args=mass_get_args,
        src_paths=mass_path,
        dest_path=dest_path
    )
    get_dagster_logger().info(f"Running shell command: {mass_get_cmd!r}")
    try:
        response = subprocess.check_output(mass_get_cmd, shell=True)
    except subprocess.CalledProcessError as e:
        get_dagster_logger().error(f"Command errored: {e}")
        raise RetryRequested(max_retries=1, seconds_to_wait=30) from e
    else:
        get_dagster_logger().info(f"Command returned response: {response}")


@op
def extract_mass_retrieval(context, mass_tar_name):
    untar_cmd_template = context.op_config["untar_cmd"]
    dest_path = context.op_config["dest_path"]
    extract_path = os.path.join(dest_path, mass_tar_name)
    untar_cmd = untar_cmd_template.format(
        path=extract_path,
        dest_root=dest_path
    )
    # XXX duplicate of content in `retrieve_from_mass()`!
    get_dagster_logger().info(f"Running shell command: {untar_cmd!r}")
    try:
        response = subprocess.check_output(untar_cmd, shell=True)
    except subprocess.CalledProcessError as e:
        get_dagster_logger().error(f"Command errored: {e}")
        raise RetryRequested(max_retries=1, seconds_to_wait=30) from e
    else:
        get_dagster_logger().info(f"Command returned response: {response}")


@op
def filter_mass_retrieval(context, mass_tar_name):
    unzip_cmd_template = context.op_config["unzip_cmd"]
    dest_path = context.op_config["dest_path"]
    extract_path = os.path.join(dest_path, mass_tar_name)
    filter_products = context.op_config["products"]
    variable_fname_template = context.op_config["variable_fname_template"]
    for product in filter_products:
        variable_fname = variable_fname_template.format(
            timestamp="*",
            product=product,
            resolution=context.op_config["variable_fname_resolution"],
            area=context.op_config["variable_fname_area"]
        )
        unzip_cmd = unzip_cmd_template.format(
            root=extract_path,
            files=variable_fname
        )
        # XXX duplicate of content in `retrieve_from_mass()`!
        get_dagster_logger().info(f"Running shell command: {unzip_cmd!r}")
        try:
            response = subprocess.check_output(unzip_cmd, shell=True)
        except subprocess.CalledProcessError as e:
            get_dagster_logger().error(f"Command errored: {e}")
            raise RetryRequested(max_retries=1, seconds_to_wait=30) from e
        else:
            get_dagster_logger().info(f"Command returned response: {response}")


@graph
def mass_retrieve_and_extract(mass_path):
    mass_tar_name = os.path.basename(mass_path)
    retrieve_from_mass(mass_path)
    extract_mass_retrieval(mass_tar_name)
    filter_mass_retrieval(mass_tar_name)



# @graph
# def mass_shell_commands(mass_path):
#     with my_tempdir_maker as mytmp:
#         retrieve_from_mass(mass_path)
#         extract_mass_retrieval()
#         move_wanted()


@job
def mass_extract():
    dates = dates_to_extract()
    paths = get_mass_paths(dates)
    paths.map(mass_retrieve_and_extract).collect()


radar_config = {
    "ops": {
        "dates_to_extract": {
            "config": {
                "datetime_str": "%Y-%m-%dT%H:%MZ",
                "archive_time_chunk": "24",
                "event_start": "2020-02-14T18:00Z",
                "event_end": "2020-02-17T18:00Z",
            }
        },
        "get_mass_paths": {
            "config": {
                "mass_root": "moose:/adhoc/projects/radar_archive/data/comp/products/composites/",
                "mass_filename_template": "{dt.year:04d}{dt.month:02d}{dt.day:02d}.tar",
            }
        },
        "retrieve_from_mass": {
            "config": {
                "mass_get_cmd": "moo get {args} {src_paths} {dest_path}",
                "mass_get_args": "-f",
                "dest_path": "./radar",
                # "dest_path": "./radar",  # for radar retrieval workflows
            }
        },
        "extract_mass_retrieval": {
            "config": {
                "untar_cmd": 'tar -xf {path} --directory {dest_root}',
                "dest_path": "./radar",
            }
        },
        "filter_mass_data": {
            "config": {
                "unzip_cmd": "gunzip {root}/{files}",
                "dest_path": "./radar",
                "products": [
                    "nimrod_ng_radar_rainrate_composite",
                    "nimrod_ng_radar_qualityproduct_composite",
                ],
                "variable_fname_template": "{timestamp}_{product}_{resolution}_{area}",
                "variable_fname_resolution": "1km",
                "variable_fname_area": "UK",
            }
        },
    }
}

job_result = mass_extract.execute_in_process(
    run_config=radar_config
)