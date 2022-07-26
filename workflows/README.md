# workflows

Define workflows for data fetch and pre-processing tasks.

## Overview

Workflows are defined using [dagster](https://dagster.io/), a Python workflow tool. Dagster workflows are composed of four main elements:
* `asset` - an object in storage, such as a dataset, that we wish to utilise in a dagster workflow. See https://docs.dagster.io/concepts/assets/software-defined-assets.
* `op` - individual **tasks** in a workflow. See https://docs.dagster.io/concepts/ops-jobs-graphs/ops.
* `graph` - a collection of `op`s that work together to, for example, enable more complex tasks to be defined. See https://docs.dagster.io/concepts/ops-jobs-graphs/graphs.
* `job` - the definition of the dagster **workflow**. Note that `op`s and `graph`s are embedded in `job`s to construct a dagster workflow. See https://docs.dagster.io/concepts/ops-jobs-graphs/jobs.

A dagster workflow is defined as a Python script containing functions that are decorated with elements from the dagster API. These decorators define whether each function provides a `job`, `graph`, `op` or `asset` within the workflow.

The order in which `op`s and `asset`s are called within a `graph` or `job` allow the dagster workflow engine to define the Directed Acyclic Graph (DAG) to use to execute the workflow. Note that a `graph` can call other `graph`s to nest elements of the defined workflow, and that `graph`s can be called from the `job` to include them in the workflow definition.

Individual executions of a dagster workflow are configured using dagster config. This is a YAML file that provides configuration items (such as paths to `asset`s or specific fixed values for calculations) to either the whole workflow or individual `op`s within the workflow. Different config files can be used to customise a given workflow, for example for a dev run and a prod run. See https://docs.dagster.io/concepts/configuration/config-schema.

## Running a workflow

Although a dagster workflow is just a Python script, it needs to be called within dagster context to be executed as a dagster workflow. This can be done in a number of ways, including from the command-line and via dagster's web UI utility called dagit. See https://docs.dagster.io/concepts/ops-jobs-graphs/job-execution#execution for details on all of dagster's different workflow execution options.

### Command-line

You can run dagster workflows using the dagster command-line utility:

```bash
dagster job execute -f my_job.py
```

See also https://docs.dagster.io/concepts/ops-jobs-graphs/job-execution#dagster-cli.

### Dagit UI

Dagit is dagster's web UI utility. It offers a rich interface for configuring and running jobs, exploring the workflow's task graph, and tracking workflow execution in realtime. To run dagit you need to have installed dagit (`pip install dagit`) into your Python environment, then:

```bash
dagit -f my_job.py
```

If you are running the dagster workflow (and so also dagit) on a remote batch system, you also need to specify the hostname of the batch VM to be able to access dagit from a web browser elsewhere on the network (assuming the `HOSTNAME` environment variable is set):

```bash
dagit -h $HOSTNAME -f my_job.py
```

See also https://docs.dagster.io/concepts/ops-jobs-graphs/job-execution#dagit and https://docs.dagster.io/concepts/dagit/dagit.