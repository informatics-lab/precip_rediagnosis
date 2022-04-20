"""
This script demonstrates a basic ML pipeline for the precip rediagnosis project, with a 1D convolutional model, combining vertical profile features with single level features. This script is intended for running on azureML.
"""
# core python imports
import argparse
import tempfile
import pathlib

# third party imports
import numpy as np

import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv1D, concatenate
from tensorflow.keras.layers import ZeroPadding1D, Reshape, Input, Dropout, PReLU
from tensorflow.keras.models import Sequential, Model

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# azure specific imports
import azureml.core


def build_model(nprof_features, nheights, nsinglvl_features):
    """
    TODO: add documentation on the building of the model

    """
    #  TODO: should some of these parameters be specified somewhere centrally?
    profile_input = Input(shape=(nheights, nprof_features), name='profile_input')
    prof_size = nheights*nprof_features

    out = ZeroPadding1D(padding=1)(profile_input)
    out = Conv1D(32, 3, strides=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')(out)
    ident = out
    out = ZeroPadding1D(padding=1)(out)
    out = Conv1D(32, 3, strides=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')(out)
    out = ZeroPadding1D(padding=1)(out)
    out = Conv1D(32, 3, strides=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')(out)
    x = tf.keras.layers.add([out, ident])
    out = Flatten()(x)
    out = Dense(prof_size, use_bias=False, activation='relu')(out)

    if nsinglvl_features > 0:
        surf_input = Input(shape=(nsinglvl_features,), name='surf_input')
        flat_profs = Flatten()(profile_input)
        raw_in = tf.keras.layers.concatenate([flat_profs, surf_input])
        raw_size = (nheights*nprof_features)+nsinglvl_features
        
        out = tf.keras.layers.concatenate([out, surf_input])
        x = tf.keras.layers.add([out, raw_in])
        x = Dense(1024, use_bias=False, activation='relu')(x)
        x = Dense(1024, use_bias=False, activation='relu')(x)
        
        main_output = Dense(1, use_bias=True, activation='linear', name='main_output')(x)
        model = Model(inputs=[profile_input, surf_input], outputs=[main_output])
    
    else:
        main_output = Dense(1, use_bias=True, activation='linear', name='main_output')(out)
        model = Model(inputs=[profile_input], outputs=[main_output])
        
    return model

def train_model(model, data_splits):
    # TODO: these hyperparameters should be read in from somewhere?
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_absolute_error', optimizer=optimizer)

    history = model.fit(data_splits['X_train'], 
                        data_splits['y_train'], 
                        epochs=1, 
                        batch_size=32, 
                        validation_split=0.25, verbose=True)
    return model
    
def load_data(current_ws, dataset_name):
    dataset = azureml.core.Dataset.get_by_name(current_ws, name=dataset_name)
    input_data = dataset.to_pandas_dataframe()    
    return input_data

def preprocess_data(input_data, feature_dict):
    # drop NaN values in the dataset
    
    data = input_data.dropna()
    data = data[data[feature_dict['target']]>0]

    # Get a list of columns names for profile features
    prof_feature_columns = [s for s in data.columns for vars in feature_dict['profile'] if s.startswith(vars)]

    print(prof_feature_columns )
    print(feature_dict['single_level'])
    features = data[prof_feature_columns + feature_dict['single_level']]
    print(features)
    
    target = data[[feature_dict['target']]]
    # drop data points with zero precip in the radar data
    
    standardScaler = StandardScaler()

    features = pd.DataFrame(standardScaler.fit_transform(features), 
                                columns=features.columns,
                                index=features.index)
    processed_data = pd.concat([features, target], axis=1, sort=False)

    # Height profiles data
    X_train_prof, X_test_prof, y_train, y_test = train_test_split(
        features[prof_feature_columns],
        target,
        test_size=0.2,
        random_state=1
    )

    # Single level data
    X_train_singlvl, X_test_singlvl, y_train, y_test = train_test_split(
        features[feature_dict['single_level']],
        target,
        test_size=0.2,
        random_state=1
    )

    # reshape height profile variables 
    X_train_prof = np.transpose(X_train_prof.to_numpy().reshape(X_train_prof.shape[0], 2, 33), (0, 2, 1))
    X_test_prof = np.transpose(X_test_prof.to_numpy().reshape(X_test_prof.shape[0], 2, 33), (0, 2, 1))
    # y_test and y_train is the same in both of these, given that the random state is set    
    
    data_dims_dict = {
        'nprof_features' : len(feature_dict['profile']),
        'nheights' : len(prof_feature_columns)//len(feature_dict['profile']),
        'nsinglvl_features' :len(feature_dict['single_level']),
    }
    
    if data_dims_dict['nsinglvl_features'] > 0:
        X_train = [X_train_prof, X_train_singlvl]
        X_test = [X_test_prof, X_test_singlvl]
    else:
        X_train = X_train_prof
        X_test = X_test_prof  
        
    data_splits = {'X_train': X_train,
                   'X_test': X_test,
                   'y_train': y_train,
                   'y_test' : y_test,
                  }
    return data_splits, data_dims_dict

def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset-name', dest='dataset_name',
                        help='the name of the azure dataset to load.')
    
    parser.add_argument('--target-parameter', dest='target_parameter')
    parser.add_argument('--profile-features', dest='profile_features',nargs='*')
    parser.add_argument('--single-level_features', dest='single_level_features',nargs='*')
    parser.add_argument('--model-name', dest='model_name')
                 
    args = parser.parse_args()
    return args
        
    
def main():
    
    args = get_args()
    
    feature_dict = {'profile': args.profile_features,
                    'single_level': args.single_level_features,
                    'target': args.target_parameter,
                   }
    print(feature_dict)
    
    prd_run = azureml.core.Run.get_context()
    
    # We dont access a workspace in the same way in a script compared to a notebook, as described in the stackoverflow post:
    # https://stackoverflow.com/questions/68778097/get-current-workspace-from-inside-a-azureml-pipeline-step 
    prd_ws = prd_run.experiment.workspace

    print(sklearn.__version__)
    input_data = load_data(prd_ws, args.dataset_name)
    data_splits, data_dims = preprocess_data(input_data, feature_dict)

    model = build_model(**data_dims)

    model = train_model(model, data_splits)

    y_pred = model.predict(data_splits['X_test'])
    
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
    ma_error = mean_absolute_error(data_splits['y_test'], y_pred)
    prd_run.log('MAE', ma_error)
    rsqrd = r2_score(data_splits['y_test'], y_pred)
    prd_run.log(f'R-squared score', rsqrd)
    


main()