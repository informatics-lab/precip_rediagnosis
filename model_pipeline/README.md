# Model pipeline 
This directory contains scripts and notebook from initial ML pipeline development for developing a pipeline to train a model to predict precipitation rates. Initial work focused on predicting mean precipitation in a grid box as easier way to get set up, while subsequent work has focused on modelling fraction of a grid box where precipitation rates is in one of a specified list of precip intensity bands.

## Directory Contents

### Key Notebooks
* Model Training Notebooks
* [fraction_model_dev_reduced_bands.ipynb ](fraction_model_dev_reduced_bands.ipynb ) - Training a model for predicting fractions in precip intensity_bands.
  * [fraction_model_dev_reduced_bands_azml.ipynb](fraction_model_dev_reduced_bands_azml.ipynb)  - Same noteboook running on AzureML
* ML Ops notebooks
  * [prd_mlops_tracking_server.ipynb](prd_mlops_tracking_server.ipynb) - Demonstrating use an MLOps server for tracking experiments and runs.
* Notebooks demonstrating how to run on AzureML
  * [prd_mlops_azml_cluster_demo.ipynb](prd_mlops_azml_cluster_demo.ipynb) - Demomnstrating how to use the script below to launch training on AzureML cluster.
  * [prd_mlops_azml_cluster_hyperdrive_demo.ipynb](prd_mlops_azml_cluster_hyperdrive_demo.ipynb) - Demonstrating how to use hyperdrive on an AzureML cluster to do hyperparameter tuning in parallel.
  * [prd_mlflow_on_azure_demo.ipynb] - Notebook demonstrating how to use the ML Flow API to report experiment and run details to the AzureML backend.


### Key Scripts

* [prd_pipeline.py](prd_pipeline.py) - Notebook containing key classes for running training on an AzureML compute cluster, including loading and preparing data and building and training the model.
* [prd_cluster_train_demo.py](prd_cluster_train_demo.py) - Entry point script for running training (a single trial) on an azureML cluster.
