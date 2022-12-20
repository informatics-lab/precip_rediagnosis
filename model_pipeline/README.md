# Precipitation rediagnosis model pipeline
This directory contains developments of a machine learning model to predict precipitation, to be specific it is predicting the fraction of each MOGREPS-G grid cell that falls into a set of instantaneous precipitation intensity bands for a region over the UK. 

A number of demo notebooks have been produced to demonstrate the various aspects of the model training and evaluation step of the machine learning pipeline. 

### Model training on SPICE
- [Model training on SPICE](prd_demo_fraction_model_spice.ipynb) - This notebook demonstrates the process of training a tensorflow 1D Convolutional Neural Network outputting fractions in a set of classes (precipitation intensity bands). MLFlow is used within the notebook to track model training.
- [Rebalancing incorporated into model training](prd_demo_fraction_model_rebalancing_spice.ipynb) - This notebook is very similiar to the notebook above, however demonstrates the use of technique for data rebalancing during preprocessing using under- and over-sampling in order to reduce the impact of class imbalance during model training.

### Model training and optimization on AzureML
- [Model training on AzureML](prd_demo_fraction_model_azml.ipynb) - This notebook again demonstrates the model training pipeline, however is designed to be run on AzureML. 
- [Automated hyperparameter on AzureML](prd_mlops_azml_cluster_hyperdrive_demo_fractions.ipynb) - This notebook demonstrates the use of <i>hyperdrive</i>, an AzureML tool for automated hyperparameter tuning. This notebook also contains a demonstration of how model training can be run on an AzureML cluster.

### Inference and evaluation
- [visualisation of predictions](prd_demo_test_prediction_visualisation.ipynb) - This notebook demonstrates how to load in a trained model from a MLFlow experiment run and apply it to test data for inference. The resulting model predictions are then visualised. 
- [model evaluation demo](prd_demo_model_evaluation_test_scenarios.ipynb) - This notebook demonstrates the evaluation of a pretrained model on unseen test data using fractional skill score. This also loads a pretrained model from an MLFlow experiment run. 
- [Explainable AI demo](prd_demo_XAI.ipynb) - This notebook demonstrates the application of explainable AI techniques for understanding the predictions made by the machine learning model. The techniques demonstrate here are: filter visualisation, permutation feature importance and saliency maps. 
