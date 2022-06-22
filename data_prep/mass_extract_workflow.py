import datetime
import glob
import os
import subprocess
from tempfile import TemporaryDirectory

from dagster import (
    get_dagster_logger,
    graph, job, op,
    make_values_resource,
    DynamicOut, DynamicOutput, Out,
    RetryRequested,
)
import pandas as pd


@op(required_resource_keys={"setup"})
def run_cmd(context, cmd):
    sep = context.resources.setup["sep"]
    get_dagster_logger().info(f"Running shell command: {cmd!r}")
    try:
        response = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        get_dagster_logger().error(f"Command errored: {e}")
        raise RetryRequested(max_retries=1, seconds_to_wait=30) from e
    else:
        get_dagster_logger().info(f"Command returned response: {response!r}")
    return f"{cmd}{sep}{response}"


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


@op(required_resource_keys={"setup"}, out=DynamicOut())
def get_mass_paths(context, dates):
    mass_root = context.resources.setup["mass_root"]
    mass_filename_template = context.op_config["mass_filename_template"]
    for i, dt in enumerate(dates):
        mass_path = os.path.join(mass_root, mass_filename_template.format(dt=dt))
        yield DynamicOutput(mass_path, mapping_key=f"path_{i}")


@op(required_resource_keys={"setup"})
def retrieve_from_mass(context, mass_fname, temp_dir) -> str:
    """Get a file from MASS."""
    mass_root = context.resources.setup["mass_root"]
    mass_get_cmd_template = context.op_config["mass_get_cmd"]
    dest_path = os.path.join(temp_dir.name, context.resources.setup["dest_path"])
    mass_path = os.path.join(mass_root, mass_fname)
    mass_get_cmd = mass_get_cmd_template.format(
        args=context.op_config["mass_get_args"],
        src_paths=mass_path,
        dest_path=dest_path
    )
    return mass_get_cmd


@op(required_resource_keys={"setup"})
def extract_mass_retrieval(context, temp_dir, mass_fname, _) -> str:
    """Extract a tar archive file retrieved from MASS."""
    untar_cmd_template = context.op_config["untar_cmd"]
    dest_path = context.resources.setup["dest_path"]
    extract_path = os.path.join(temp_dir.name, dest_path)
    mass_tar_name = os.path.basename(mass_fname)
    extracted_tar_name = os.path.join(extract_path, mass_tar_name)
    untar_cmd = untar_cmd_template.format(
        path=extracted_tar_name,
        dest_root=extract_path
    )
    return untar_cmd


@op(required_resource_keys={"setup"}, out=DynamicOut())
def filter_mass_retrieval(context, temp_dir, _):
    """Filter the extracted archive from MASS for specific files of interest."""
    filter_products = context.op_config["products"]
    variable_fname_template = context.op_config["variable_fname_template"]
    gunzip_files = []
    for product in filter_products:
        variable_fname = variable_fname_template.format(
            timestamp=context.op_config["variable_fname_timestamp"],
            product=product,
            resolution=context.op_config["variable_fname_resolution"],
            area=context.op_config["variable_fname_area"]
        )
        gunzip_files.extend(glob.glob(os.path.join(temp_dir.name, variable_fname)))

    unzip_cmd_template = context.op_config["unzip_cmd"]
    dest_root = context.resources.setup["retrieve_path_root"]
    for i, zip_file in enumerate(gunzip_files):
        # XXX this won't work if our files have a file extension..?
        filename = os.path.splitext(os.path.basename(zip_file))[0]
        dest_path = os.path.join(dest_root, filename)
        unzip_cmd = unzip_cmd_template.format(
            zip_file=zip_file,
            dest_path=dest_path
        )
        yield DynamicOutput(unzip_cmd, mapping_key=f"unzip_{i}")


@op(required_resource_keys={"setup"})
def logit(context, responses):
    """Record important output from cmd executions to a log file."""
    sep = context.resources.setup["sep"]
    logdir = context.resources.setup["logdir"]
    logfile = context.resources.setup["logfile"]
    logfilename = os.path.join(logdir, logfile)
    if isinstance(responses, str):
        responses = [responses]
    for response in responses:
        cmd, _ = response.split(sep)
        path = cmd.split(" ")[-1]
        get_dagster_logger().info(path)
    return "done!"


@op(required_resource_keys={"setup"})
def create_temp_dir(context):
    retrieve_path_root = context.resources.setup["retrieve_path_root"]
    tempdir = TemporaryDirectory(prefix=retrieve_path_root)
    get_dagster_logger().info(f"Temporary extract directory: {tempdir.name}")
    return tempdir


@op
def remove_temp_dir(tempdir, _):
    get_dagster_logger().info(f"Temporary directory {tempdir.name} removed")
    tempdir.cleanup()


@graph
def mass_retrieve_and_extract(mass_fname):
    tempdir = create_temp_dir()
    retrieve_resp = run_cmd(retrieve_from_mass(mass_fname, tempdir))
    _ = logit(retrieve_resp)
    extract_resp = run_cmd(extract_mass_retrieval(tempdir, mass_fname, retrieve_resp))
    unzip_cmds = filter_mass_retrieval(tempdir, extract_resp)
    resps = unzip_cmds.map(run_cmd)
    done = logit(resps.collect())
    remove_temp_dir(tempdir, done)


@job(
    resource_defs={"setup": make_values_resource(
        mass_root=str,
        retrieve_path_root=str,
        dest_path=str,
        sep=str,
        logdir=str,
        logfile=str,
    )}
)
def mass_extract():
    dates = dates_to_extract()
    paths = get_mass_paths(dates)
    paths.map(mass_retrieve_and_extract)
    # results.collect()


# job_result = mass_extract.execute_in_process(
#     run_config=radar_config
# )