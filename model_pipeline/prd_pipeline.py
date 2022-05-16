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

    print(feature_dict)
    features = data[prof_feature_columns + feature_dict['single_level']]
    
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

def calc_metrics(current_run, data_splits, y_pred):
    metrics_dict = {}
    metrics_dict['mean_absolute_error'] = mean_absolute_error(data_splits['y_test'], y_pred)
    metrics_dict['R-squared score'] = r2_score(data_splits['y_test'], y_pred)
    for k1, v1 in metrics_dict.items():
        current_run.log(k1, v1)
    return metrics_dict


        
    