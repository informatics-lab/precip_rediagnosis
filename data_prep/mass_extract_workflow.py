import datetime
import glob
import os
import shutil
import subprocess
from tempfile import mkdtemp

from dagster import (
    In, Nothing, Out,
    get_dagster_logger,
    graph, job, op, resource,
    make_values_resource,
    DynamicOut, DynamicOutput,
    Backoff, Jitter, RetryPolicy, RetryRequested,
    failure_hook, HookContext,
)
import pandas as pd


class EncapsulatedMkdTemp(object):
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
        return self.tempdir

    def mk_tempdir(self):
        self.tempdir = mkdtemp(prefix=self.prefix)

    def rm_tempdir(self):
        shutil.rmtree(self.name)


@resource(required_resource_keys={"setup"})
def tempdir_resource(context):
    prefix = context.resources.setup["retrieve_path_root"]
    tempdir = EncapsulatedMkdTemp(prefix)
    get_dagster_logger().info(f"Temporary extract directory: {tempdir.name}")
    return tempdir


@op(required_resource_keys={"setup"})
def make_dest_root(context):
    dest_root = context.resources.setup["retrieve_path_root"]
    try:
        os.mkdir(dest_root, mode=0o755)
    except FileExistsError:
        get_dagster_logger().info(f"Directory {dest_root!r} already exists; nothing to do.")
    else:
        get_dagster_logger().info(f"Created directory {dest_root!r}.")


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
    mass_get_cmd_template = context.op_config["mass_get_cmd"]
    mass_get_cmd = mass_get_cmd_template.format(
        args=context.op_config["mass_get_args"],
        src_paths=mass_fname,
        dest_path=temp_dir.name
    )
    return mass_get_cmd


@op
def extract_mass_retrieval(context, temp_dir, mass_fname, _) -> str:
    """Extract a tar archive file retrieved from MASS."""
    untar_cmd_template = context.op_config["untar_cmd"]
    mass_tar_name = os.path.basename(mass_fname)
    extracted_tar_name = os.path.join(temp_dir.name, mass_tar_name)
    untar_cmd = untar_cmd_template.format(
        path=extracted_tar_name,
        dest_root=temp_dir.name
    )
    return untar_cmd


@op(required_resource_keys={"setup"})
def filter_mass_retrieval(context, temp_dir, _) -> list:
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
    unzip_cmds = [temp_dir]
    for zip_file in gunzip_files:
        # XXX this won't work if our files have a file extension..?
        filename = os.path.splitext(os.path.basename(zip_file))[0]
        dest_path = os.path.join(dest_root, filename)
        unzip_cmd = unzip_cmd_template.format(
            zip_file=zip_file,
            dest_path=dest_path
        )
        unzip_cmds.append(unzip_cmd)
    return unzip_cmds


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


@op(required_resource_keys={"tempdir_resource"})
def create_temp_dir(context, _):
    tempdir = context.resources.tempdir_resource
    get_dagster_logger().info(f"Temporary directory created: {tempdir.name!r}")
    return tempdir


@op(ins={"start": In(Nothing)})
def remove_temp_dir(tempdir):
    get_dagster_logger().info(f"Temporary directory {tempdir.name} removed")
    tempdir.rm_tempdir()


@failure_hook(required_resource_keys={"setup"})
def on_fail_remove_tempdirs(context: HookContext):
    dest_root = context.resources.setup["retrieve_path_root"]
    tempdirs = list(os.walk(dest_root))[0][1]
    if len(tempdirs):
        for tempdir in tempdirs:
            shutil.rmtree(os.path.join(dest_root, tempdir))
        get_dagster_logger().info(f"Temporary directories removed: {tempdirs}")
    else:
        get_dagster_logger().info("No temporary directories found; nothing to clean up.")


@graph
def mass_retrieve_and_extract(mass_fname):
    temp_dir = create_temp_dir(mass_fname)
    retrieve_resp = run_cmd(retrieve_from_mass(temp_dir, mass_fname))
    logit(retrieve_resp)
    extract_resp = run_cmd(extract_mass_retrieval(temp_dir, mass_fname, retrieve_resp))
    unzip_cmds_list = filter_mass_retrieval(temp_dir, extract_resp)
    return unzip_cmds_list


@op(out={"tempdirs_out": Out(), "cmds_out": Out()})
def gather(cmds):
    """
    Gather lists of collected results from dynamic graph execution.
    Also sort the results, as the results list has been overloaded to
    contain the temporary directory name as the 0th item in the list.

    """
    get_dagster_logger().info(f"Gather received: {cmds}")
    tempdirs = []
    unzip_cmds = []
    for cmd_list in cmds:
        tempdir, *unzip_cmds = cmd_list
        tempdirs.append(tempdir)
        unzip_cmds.extend(unzip_cmds)
    return (tempdirs, unzip_cmds)


@op(out=DynamicOut())
def dynamicise(l):
    """Convert an ordinary Python iterable `l` into a dagster dynamic output."""
    for i, l_itm in enumerate(l):
        yield DynamicOutput(l_itm, mapping_key=f"unzip_{i}")


@job(
    hooks={on_fail_remove_tempdirs},
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
    make_dest_root()
    dates = dates_to_extract()
    paths = get_mass_paths(dates)
    graph_output = paths.map(mass_retrieve_and_extract)
    tempdirs_list, unzip_cmds_list = gather(graph_output.collect())
    unzip_cmds = dynamicise(unzip_cmds_list)
    unzip_retry_policy = RetryPolicy(
        delay=5, backoff=Backoff.EXPONENTIAL, jitter=Jitter.PLUS_MINUS
    )
    results = unzip_cmds.map(run_cmd.with_retry_policy(unzip_retry_policy))
    tempdirs = dynamicise(tempdirs_list)
    tempdirs.map(lambda tmp: remove_temp_dir(tmp, start=logit(results.collect())))