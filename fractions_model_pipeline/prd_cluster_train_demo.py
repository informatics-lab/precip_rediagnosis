"""
This script demonstrates a basic ML pipeline for the precip rediagnosis project, with a 1D convolutional model, combining vertical profile features with single level features. This script is intended for running on azureML.
"""


# core python imports
import argparse
import tempfile
import pathlib
import pandas as pd
import numpy as np
import tensorflow as tf

# NB No AzreML imports in this file!!!

import prd_pipeline


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset-name', dest='dataset_name',
                        help='the name of the azure dataset to load.')
    
    parser.add_argument('--target-parameter', dest='target_parameter', nargs='*')
    parser.add_argument('--profile-features', dest='profile_features',nargs='*')
    parser.add_argument('--single-level_features', dest='single_level_features',nargs='*')
    parser.add_argument('--model-name', dest='model_name')
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--batch-size', dest='batch_size', type=int)
    parser.add_argument('--learning-rate', dest='learning_rate', type=float)
    parser.add_argument('--test-fraction', dest='test_fraction', type=float)
    parser.add_argument('--log-dir', dest='log_dir')
    parser.add_argument('--data-path',dest='data_path')
    parser.add_argument('--blob',dest='from_blobstore',action='store_true')
    parser.add_argument('--autolog',dest='autolog',action='store_true')    

    args = parser.parse_args()
    return args


def main():
    
    args = get_args()
    
    if args.autolog:
        prd_pipeline.setup_logging()

    print(args.profile_features)
    feature_dict = {
        'profile': args.profile_features,
        'single_level': args.single_level_features,
        'target': args.target_parameter
    }    
    
    if args.data_path is not None:
        if args.from_blobstore:
            with open('credentials_file.json') as credentials_file:
                az_blob_cred = json.load(credentials_file)
            input_data = prd_pipeline.load_data_azure_blob(az_blob_cred, args.data_path)
        else:
            input_data = prd_pipeline.load_data_local(args.data_path)
    else:
        input_data = prd_pipeline.load_data_azml_dataset(args.dataset_name)
        
    df_train, df_test = prd_pipeline.random_time_space_sample(
        input_data, test_fraction=args.test_fraction, random_state=np.random.RandomState(), sampling_columns = ['time', 'latitude', 'longitude'])
    data_splits, data_dims_dict = prd_pipeline.preprocess_data(
        df_train, feature_dict, test_fraction=args.test_fraction/(1-args.test_fraction))
    
    model = prd_pipeline.build_model(**data_dims_dict)
    
    hyperparameter_dict = {
        'epochs': args.epochs, 
        'learning_rate': args.learning_rate, 
        'batch_size': args.batch_size,
        'class_weights': None, 
        'loss': tf.keras.losses.KLDivergence()
    }
    
    model, history = prd_pipeline.train_model(model, data_splits, hyperparameter_dict, log_dir=args.log_dir)
    
    y_pred = model.predict(data_splits['X_val'])
    
    prd_model_name = args.model_name
    
    prd_pipeline.save_model(model, prd_model_name)
    
main()
