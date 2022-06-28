import atexit
import datetime
import glob
from hashlib import blake2b
import os
import shutil
import subprocess
from tempfile import TemporaryDirectory, mkdtemp
from typing import List

from dagster import (
    In, Nothing,
    get_dagster_logger,
    graph, job, op, resource,
    make_values_resource,
    DynamicOut, DynamicOutput,
    RetryRequested,
)
import pandas as pd


class EncapsulatedTemporaryDir(object):
    def __init__(self, prefix):
        self.prefix = prefix
        if not self.prefix.endswith(os.sep):
            self.prefix += os.sep

        self._tempdir = None
        self.mk_tempdir()

    @property
    def tempdir(self):
        if self._tempdir is None:
            self.mk_tempdir()
        return self._tempdir

    @tempdir.setter
    def tempdir(self, value):
        self._tempdir = value

    @property
    def name(self):
        return self.tempdir.name

    def mk_tempdir(self):
        self.tempdir = TemporaryDirectory(prefix=self.prefix)

    def rm_tempdir(self):
        self.tempdir.cleanup()


class EncapsulatedMkdTemp(object):
    def __init__(self, prefix):
        self.prefix = prefix
        if not self.prefix.endswith(os.sep):
            self.prefix += os.sep

        self._tempdir = None
        self.mk_tempdir()

        # Automatic cleanup on process exit.
        # atexit.register(self.rm_tempdir)

    @property
    def tempdir(self):
        if self._tempdir is None:
            self.mk_tempdir()
        return self._tempdir

    @tempdir.setter
    def tempdir(self, value):
        self._tempdir = value

    @property
    def name(self):
        return self.tempdir

    def mk_tempdir(self):
        self.tempdir = mkdtemp(prefix=self.prefix)

    def rm_tempdir(self):
        shutil.rmtree(self.name)


@resource(required_resource_keys={"setup"})
def tempdir_resource(context):
    prefix = context.resources.setup["retrieve_path_root"]
    # tempdir = EncapsulatedTemporaryDir(prefix)
    tempdir = EncapsulatedMkdTemp(prefix)
    get_dagster_logger().info(f"Temporary extract directory: {tempdir.name}")
    return tempdir


@op(required_resource_keys={"setup"})
def run_cmd(context, cmd: str):
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
def retrieve_from_mass(context, temp_dir, mass_fname) -> str:
    """Get a file from MASS."""
    # temp_dir = context.resources.tempdir_resource
    mass_root = context.resources.setup["mass_root"]
    mass_get_cmd_template = context.op_config["mass_get_cmd"]
    # dest_path = os.path.join(temp_dir.name, context.resources.setup["dest_path"])
    mass_path = os.path.join(mass_root, mass_fname)
    mass_get_cmd = mass_get_cmd_template.format(
        args=context.op_config["mass_get_args"],
        src_paths=mass_fname,
        dest_path=temp_dir.name
    )
    return mass_get_cmd


@op
def extract_mass_retrieval(context, temp_dir, mass_fname, _) -> str:
    """Extract a tar archive file retrieved from MASS."""
    # temp_dir = context.resources.tempdir_resource
    untar_cmd_template = context.op_config["untar_cmd"]
    # dest_path = context.resources.setup["dest_path"]
    # extract_path = os.path.join(temp_dir.name, dest_path)
    mass_tar_name = os.path.basename(mass_fname)
    extracted_tar_name = os.path.join(temp_dir.name, mass_tar_name)
    untar_cmd = untar_cmd_template.format(
        path=extracted_tar_name,
        dest_root=temp_dir.name
    )
    return untar_cmd


@op(required_resource_keys={"setup"}, out=DynamicOut())
def filter_mass_retrieval(context, temp_dir, _) -> List[DynamicOutput[str]]:
    """Filter the extracted archive from MASS for specific files of interest."""
    # temp_dir = context.resources.tempdir_resource
    filter_products = context.op_config["products"]
    # get_dagster_logger().info(f"Temp directory: {temp_dir.name}")
    # get_dagster_logger().info(f"Filter products: {filter_products}")
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
    # get_dagster_logger().info(f"First 10 files to unzip: {gunzip_files[:10]} ...")

    unzip_cmd_template = context.op_config["unzip_cmd"]
    dest_root = context.resources.setup["retrieve_path_root"]
    unzip_cmds = []
    for zip_file in gunzip_files:
        # XXX this won't work if our files have a file extension..?
        filename = os.path.splitext(os.path.basename(zip_file))[0]
        dest_path = os.path.join(dest_root, filename)
        unzip_cmd = unzip_cmd_template.format(
            zip_file=zip_file,
            dest_path=dest_path
        )
        i = blake2b(salt=os.urandom(blake2b.SALT_SIZE)).hexdigest()[:16]
        unzip_cmds.append((f"unzip_{i}", unzip_cmd))
        # yield DynamicOutput(unzip_cmd, mapping_key=f"unzip_{i}")
    get_dagster_logger().info(f"Unzip commands: {[cmd for (_, cmd) in unzip_cmds]}")
    return [DynamicOutput(cmd, mapping_key=i) for (i, cmd) in unzip_cmds[:10]]


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


@op
def dir_check(tempdir):
    import time
    get_dagster_logger().info(f"Temporary directory: {tempdir.name!r}")
    time.sleep(30)


@op(required_resource_keys={"tempdir_resource"})
def create_temp_dir(context, _):
    tempdir = context.resources.tempdir_resource
    get_dagster_logger().info(f"Temporary directory created: {tempdir.name!r}")
    return tempdir


@op(ins={"start": In(Nothing)})
def remove_temp_dir(tempdir):
    # XXX this must always run!
    get_dagster_logger().info(f"Temporary directory {tempdir.name} removed")
    tempdir.rm_tempdir()


@graph
def mass_retrieve_and_extract(mass_fname):
    temp_dir = create_temp_dir(mass_fname)
    retrieve_resp = run_cmd(retrieve_from_mass(temp_dir, mass_fname))
    logit(retrieve_resp)
    extract_resp = run_cmd(extract_mass_retrieval(temp_dir, mass_fname, retrieve_resp))
    unzip_cmds = filter_mass_retrieval(temp_dir, extract_resp)
    resps = unzip_cmds.map(run_cmd)
    # logit(resps.collect())
    remove_temp_dir(temp_dir, start=logit(resps.collect()))
    # remove_temp_dir(temp_dir, start=dir_check(temp_dir))


@job(
    resource_defs={
        "setup": make_values_resource(
            mass_root=str,
            retrieve_path_root=str,
            dest_path=str,
            sep=str,
            logdir=str,
            logfile=str,
        ),
        "tempdir_resource": tempdir_resource,
    }
)
def mass_extract():
    dates = dates_to_extract()
    paths = get_mass_paths(dates)
    paths.map(mass_retrieve_and_extract)
