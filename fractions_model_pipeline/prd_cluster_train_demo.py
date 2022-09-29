"""
This script demonstrates a basic ML pipeline for the precip rediagnosis project, with a 1D convolutional model, combining vertical profile features with single level features. This script is intended for running on azureML.
"""


# core python imports
import argparse
import tempfile
import pathlib
import pandas as pd
import numpy as np

# azure specific imports
import azureml.core

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

    args = parser.parse_args()
    return args


def main():
    
    args = get_args()
    
    print(args.profile_features)
    feature_dict = {
        'profile': args.profile_features,
        'single_level': args.single_level_features,
        'target': args.target_parameter
    }    
    
    prd_run = azureml.core.Run.get_context()
    
    # We dont access a workspace in the same way in a script compared to a notebook, as described in the stackoverflow post:
    # https://stackoverflow.com/questions/68778097/get-current-workspace-from-inside-a-azureml-pipeline-step 
    prd_ws = prd_run.experiment.workspace
    
    # LOAD CURRENTLY NOT WORKING  
    # input_data = prd_pipeline.load_data(prd_ws, args.dataset_name)
    # data_splits, data_dims_dict = prd_pipeline.preprocess_data(
    #     input_data, feature_dict, 
    #     test_fraction=args.test_frac, test_savefn=args.test_filename)
    
    # WORK AROUND...
    train202208_dataset_all = azureml.core.Dataset.get_by_name(prd_ws, name=args.dataset_name)
    with train202208_dataset_all.mount() as mount1:
        prd_path_list = [p1 for p1 in pathlib.Path(mount1.mount_point).rglob('*csv') ]
        merged_df = pd.concat([pd.read_csv(p1) for p1 in prd_path_list])  
        
    df_train, df_test = prd_pipeline.random_time_space_sample(
        merged_df, test_fraction=args.test_fraction, random_state=np.random.RandomState(), sampling_columns = ['time', 'latitude', 'longitude'])
    data_splits, data_dims_dict = prd_pipeline.preprocess_data(
        df_train, feature_dict, test_fraction=args.test_fraction/(1-args.test_fraction))
    
    model = prd_pipeline.build_model(**data_dims_dict)
    
    hyperparameter_dict = {
        'epochs': args.epochs, 
        'learning_rate': args.learning_rate, 
        'batch_size': args.batch_size
    }
    
    model, history = prd_pipeline.train_model(model, data_splits, hyperparameter_dict, log_dir=args.log_dir)
    
    # for k1, v1 in model.history.history.items():
    for k1, v1 in history.history.items():
        prd_run.log(k1, v1[-1])

    y_pred = model.predict(data_splits['X_val'])
    
    prd_model_name = args.model_name
    
    # save the model to temp directory and save to the run. 
    # (The local files will be cleaned up with the temp directory.)
    with tempfile.TemporaryDirectory() as td1:
        # save model architecure as JSON
        # this can be loaded using tf.keras.models.model_from_json and then training can be run
        model_json_path = pathlib.Path(td1) / (prd_model_name + '.json')
        with open(model_json_path,'w') as json_file:
            json_file.write(model.to_json())
        prd_run.upload_file(name=prd_model_name + '_architecture', path_or_stream=str(model_json_path) )
        model_save_path = pathlib.Path(td1) / prd_model_name
        model.save(model_save_path)
        prd_run.upload_folder(name=prd_model_name, path=str(model_save_path))
        prd_run.register_model(prd_model_name, prd_model_name + '/')
    # calculate some metrics
    # ma_error, rsqd = prd_pipeline.calc_metrics(prd_run, data_splits, y_pred)

main()
