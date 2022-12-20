# Data Preparation
The contents of this directory are all about extracting and preprocessing data from the Met Office archive and other sources for the precipitation rediagnosis. In general we are seeking to match ensemble model output  with assorted sources of ground truth  for UK rainfall.

Current data sources:
* [UK Met Office MOGREPS-G ensemble](https://www.metoffice.gov.uk/research/weather/ensemble-forecasting/mogreps)
* [UK Met Office Rainfall Radar](https://www.metoffice.gov.uk/public/weather/observation/rainfall-radar)


## Description of contents
The contents is dividided into different sorts of material, each sort is described in its own section below.

### Key notebooks 
Jupyter notebooks are used to demonstrate techniques, tools and algorithms used for the data prep. Key notebooks include:
* [Data preparation demo](transform_and_preprocess_spice.ipynb) - This notebook demonstrates the steps required to prepare the merged dataset with radar and mogeps-g model data in a single data frame, including calculation of precip intensity band fractions and regridding of radar data onto the mogreps-g grid.
* [Data verification demo](prd_data_verification.ipynb) - Thisnotebook is to be run after producing a new dataset using the batch processing code (see below) to check that valid output has been produced.
  * [Data verification for azureml](azureml/prd_data_verification_azureml.ipynb) - The same notebook but for running on AzureML and loading data as required for that platform.
* [AzureML dataset creation](azureml/prd_create_azml_dataset.ipynb) - A notebook to create AzureML datastore and dataset assets which make data access for subsequent machine learning pipelines. The notebook finds all the events in a specified storage location and creates pointers to the corresponding data assets

### Batch processing files
While the above notebook present an annotated version of the key algorithms, the actual processing of data is done in a series of scripts. These are designed to be extensible so further work can add data sources and processes without disturbing existing code. 

* [extract_data.py] - The main entry point for running the data extraction and preparation steps
* [drivers.py] - Contains classes that derive from a `MassExtractor` parent class, which primarily have `extract()` and `prepare()` which prepares a dataframe from a particular data source. 
* [event_configs] - A directory containing descriptions of events for which precip data is to be extracted. Each event is described in a json file which includes general info such as the start and end dates of the event, as well as a description of each source to be extracted.
  *Currently these are primarily `ModelStageExtractor` which can process model data produced by stage from archive, and `RadarExtractor` which can process archived radar data.
* to facilitate batch processing, which runs on the Met Office Linux cluster, there are a number of helper shell scripts:
  * [run_all_events.sh] - High level script to run processing for each of the event config json files and runs a batch job per event
  * [run_data_prep_spice.sh] - A script for running processing for a single event.
  * [submit_data_prep_spice.sh] - A helper script for running `run_data_prep_spice.sh` through the batch submission system.
  * [run_interactively.sh] - a helper script to be able to debug code by setting up the same environment as in `submit_data_prep_spice.sh`.
* There are helper scripts for moving data once it is produced:
  * [local_output_copy.sh] - A script to copy only the main output files (i.e. not intermediate files) to a target directory from the temporary location to which they were etxracted.
  * [azureml/copy_to_azure.sh] - A script to upload to Azure storage from the extraction directry

#### Additional contents
* [azureml] - Directory to contain all the code that is specific to data prep on azureml.
* [dev_notebooks] - Sundry notebooks that were used for development which may have useful code for subsequent reference and development purposes.
