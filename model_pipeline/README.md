# Model pipeline 
This directory contains scripts and notebook from initial ML pipeline development for developing a pipeline to train a model to predict precipitation rates. Initial work focused on predicting mean precipitation in a grid box as easier way to get set up, while subsequent work has focused on modelling fraction of a grid box where precipitation rates is in one of a specified list of precip intensity bands.

A number of demo notebooks have been produced to demonstrate the various aspects of the model training and evaluation step of the machine learning pipeline. 

## Directory Contents

### Key Notebooks

Model training on SPICE
* Model training on SPICE - [prd_demo_fraction_model_spice.ipynb](prd_demo_fraction_model_spice.ipynb) - This notebook demonstrates the process of training a tensorflow 1D Convolutional Neural Network outputting fractions in a set of classes (precipitation intensity bands). MLFlow is used within the notebook to track model training.
* Rebalancing incorporated into model training - [prd_demo_fraction_model_rebalancing_spice.ipynb](prd_demo_fraction_model_rebalancing_spice.ipynb) - This notebook is very similiar to the notebook above, however demonstrates the use of technique for data rebalancing during preprocessing using under- and over-sampling in order to reduce the impact of class imbalance during model training.

Model training and optimization on AzureML
* Model training on AzureML - [prd_demo_fraction_model_azml.ipynb](prd_demo_fraction_model_azml.ipynb) - This notebook again demonstrates the model training pipeline, however is designed to be run on AzureML. 
* Automated hyperparameter on AzureML - [prd_mlops_azml_cluster_hyperdrive_demo_fractions.ipynb](prd_mlops_azml_cluster_hyperdrive_demo_fractions.ipynb) - This notebook demonstrates the use of <i>hyperdrive</i>, an AzureML tool for automated hyperparameter tuning. This notebook also contains a demonstration of how model training can be run on an AzureML cluster.

Inference and evaluation
* Visualisation of predictions - [prd_demo_test_prediction_visualisation.ipynb](prd_demo_test_prediction_visualisation.ipynb) - This notebook demonstrates how to load in a trained model from a MLFlow experiment run and apply it to test data for inference. The resulting model predictions are then visualised. 
* Model evaluation demo - [prd_demo_model_evaluation_test_scenarios.ipynb](prd_demo_model_evaluation_test_scenarios.ipynb) - This notebook demonstrates the evaluation of a pretrained model on unseen test data using fractional skill score. This also loads a pretrained model from an MLFlow experiment run. 
* Explainable AI demo - [prd_demo_XAI.ipynb](prd_demo_XAI.ipynb) - This notebook demonstrates the application of explainable AI techniques for understanding the predictions made by the machine learning model. The techniques demonstrate here are: filter visualisation, permutation feature importance and saliency maps. 

MLOps 
* [prd_mlops_tracking_server.ipynb](prd_mlops_tracking_server.ipynb) - Demonstrating use an MLOps server for tracking experiments and runs.
* [prd_mlflow_on_azure_demo.ipynb](prd_mlflow_on_azure_demo.ipynb) - Notebook demonstrating how to use the ML Flow API to report experiment and run details to the AzureML backend.

### Key Scripts

* [prd_pipeline.py](prd_pipeline.py) - Notebook containing key classes for running training on an AzureML compute cluster, including loading and preparing data and building and training the model.
* [prd_cluster_train_demo.py](prd_cluster_train_demo.py) - Entry point script for running training (a single trial) on an azureML cluster.