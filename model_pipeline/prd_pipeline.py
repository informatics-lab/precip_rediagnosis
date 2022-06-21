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

import pickle


def build_model(nprof_features, nheights, nsinglvl_features):
    """
    This 1D convoluational neural network take a regression approach to predict 
    precipitation on a column-wise basis. This model takes vertical profile features 
    as it's input with the option to include single height level features if required. 
    From these features it predicts deterministic precipitation values. 
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


def train_model(model, data_splits, hyperparameter_dict):
    """
    This function trains the input model with the given data samples in data_splits. 
    Hyperparameters use when fitting the model are defined in hyperparameter_dict.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameter_dict['learning_rate'])
    model.compile(loss='mean_absolute_error', optimizer=optimizer)

    history = model.fit(data_splits['X_train'], 
                        data_splits['y_train'], 
                        epochs=hyperparameter_dict['epochs'], 
                        batch_size=hyperparameter_dict['batch_size'], 
                        validation_data=(data_splits['X_val'], data_splits['y_val']), verbose=True)
    return model
    

def load_data(current_ws, dataset_name):
    """
    This function loads data from AzureML storage and returns it as a pandas dataframe 
    """
    dataset = azureml.core.Dataset.get_by_name(current_ws, name=dataset_name)
    input_data = dataset.to_pandas_dataframe()    
    return input_data


def sample_data(features_df, target_df, test_fraction=0.2, savefn=None, random_state=None):
    """
    This function creates data samples for training and testing machine learning
    This function take two pandas dataframes as inputs:
      - features_df contains data from feature columns
      - target_df contains data from target columns
    If a filename is provided for the savefn argument the test dataset is saved
    to this file and only the training input and target data is returned. 
    If savefn is None (default) then both the train and test input and target data 
    samples are returned.
    """
    n_samples = features_df.shape[0]
    test_input = features_df.sample(
        int(n_samples*test_fraction), random_state=random_state)
    train_input = features_df[~np.isin(features_df.index, test_input.index)]
    test_target = target_df[np.isin(target_df.index, test_input.index)]
    train_target = target_df[~np.isin(target_df.index, test_input.index)]
    if savefn:
        test_dataset = pd.concat([test_input, test_target], axis=1, sort=False)
        test_dataset.to_csv(savefn)
        return train_input, train_target
    else: 
        return train_input, train_target, test_input, test_target


def reshape_profile_features(df, features, data_dims_dict):
    """
    This function reshapes vertical profile data from tabular data 
    back into data with a height dimension
    """
    prof_feature_columns = [
        s for s in df.columns for vars in features if s.startswith(vars)]
    df = df[prof_feature_columns]
    df = np.transpose(
        df.to_numpy().reshape(
            df.shape[0], 
            data_dims_dict['nprof_features'], 
            data_dims_dict['nheights']
            ),
        (0, 2, 1))
    return df


def load_test_data(test_fn, feature_dict, data_dims_dict):
    """
    This function can be used to load in test data and returns 
    input and target data ready to be used to test an ML model.
    """
    test_data = pd.read_csv(test_fn)
    with open('standardScaler.pkl', 'rb') as fin:
        standardScaler = pickle.load(fin)

    prof_feature_columns = [s for s in test_data.columns for vars in feature_dict['profile'] if s.startswith(vars)]
    features = test_data[prof_feature_columns + feature_dict['single_level']]
    y_test = test_data[feature_dict['target']]

    X_test_scaled = pd.DataFrame(
        standardScaler.fit_transform(features), 
        columns=features.columns,
        index=features.index)
    
    X_test = reshape_profile_features(
        X_test_scaled, feature_dict['profile'], data_dims_dict)
    if len(feature_dict['single_level']) > 0:
        X_test = [X_test, X_test_scaled[feature_dict['single_level']]]
    
    return X_test, y_test


def preprocess_data(input_data, feature_dict, test_fraction=0.2, test_savefn=None):
    """
    Preprocesses data ahead of ML modelling
    """

    # drop NaN values in the dataset
    data = input_data.dropna()

    # drop data points with zero precip in the radar data
    data = data[data[feature_dict['target']]>0]

    # Get a list of columns names for profile features
    prof_feature_columns = [s for s in data.columns for vars in feature_dict['profile'] if s.startswith(vars)]

    print(feature_dict)
    data_dims_dict = {
        'nprof_features' : len(feature_dict['profile']),
        'nheights' : len(prof_feature_columns)//len(feature_dict['profile']),
        'nsinglvl_features' :len(feature_dict['single_level']),
    }
    
    input_data = data[prof_feature_columns + feature_dict['single_level']]
    target_data = data[[feature_dict['target']]]
    
    random_state = np.random.RandomState()  # TO DO: how to log this in experiments!

    # Extract and save test dataset
    reduced_input_data, reduced_target_data = sample_data(
        input_data, target_data,
        test_fraction=test_fraction,
        savefn=test_savefn,
        random_state=random_state)

    # Extract and return train and validate datasets
    X_train, y_train, X_val, y_val = sample_data(
        reduced_input_data, reduced_target_data,
        test_fraction=test_fraction/(1-test_fraction),
        random_state=random_state)
    
    # Scale data to have zero mean and standard deviation of one
    standardScaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        standardScaler.fit_transform(X_train), 
        columns=X_train.columns,
        index=X_train.index)

    X_val_scaled = pd.DataFrame(
        standardScaler.transform(X_val), 
        columns=X_val.columns,
        index=X_val.index)

    with open('standardScaler.pkl', 'wb') as fout:
        pickle.dump(standardScaler, fout)

    X_train = reshape_profile_features(
        X_train_scaled, feature_dict['profile'], data_dims_dict)
    X_val = reshape_profile_features(
        X_val_scaled, feature_dict['profile'], data_dims_dict)

    if len(feature_dict['single_level']) > 0:
        X_train = [X_train, X_train_scaled[feature_dict['single_level']]]
        X_val = [X_val, X_val_scaled[feature_dict['single_level']]]

    data_splits = {'X_train': X_train,
                   'X_val': X_val,
                   'y_train': y_train,
                   'y_val' : y_val,
                  }

    return data_splits, data_dims_dict


def calc_metrics(current_run, data_splits, y_pred):
    metrics_dict = {}
    metrics_dict['mean_absolute_error'] = mean_absolute_error(data_splits['y_val'], y_pred)
    metrics_dict['R-squared score'] = r2_score(data_splits['y_val'], y_pred)
    for k1, v1 in metrics_dict.items():
        current_run.log(k1, v1)
    return metrics_dict
    