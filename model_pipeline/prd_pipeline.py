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

# when using mlflow inside a training run submitted to a compute cluster, azureml
# will set up mlflow tracking as part of the standard set up, so unlike when we run
# in a notebook, the is no need to set the tracking URI
# See this doc page for details: https://learn.microsoft.com/en-us/azure/machine-learning/v1/how-to-use-mlflow?tabs=azuremlsdk under "tracking runs running on azure machine learning".

import mlflow

# azure specific imports
try:
    import azureml.core
    USING_AZML=True
except ImportError:
    print('AzureML libraries not found, using local execution functions.')
    USING_AZML=False

import fsspec
import pickle

PRD_PREFIX = 'prd'
MERGED_PREFIX = PRD_PREFIX + '_merged'
CSV_FILE_SUFFIX = 'csv'


def setup_logging():
    mlflow.tensorflow.autolog(log_model_signatures=True)

    
def build_model(nprof_features, nheights, nsinglvl_features, nbands):
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

    if nbands == 1:
        activation = 'linear'
    else:
        activation = 'softmax'
    
    if nsinglvl_features > 0:
        surf_input = Input(shape=(nsinglvl_features,), name='surf_input')
        flat_profs = Flatten()(profile_input)
        raw_in = tf.keras.layers.concatenate([flat_profs, surf_input])
        raw_size = (nheights*nprof_features)+nsinglvl_features
        
        out = tf.keras.layers.concatenate([out, surf_input])
        x = tf.keras.layers.add([out, raw_in])
        x = Dense(1024, use_bias=False, activation='relu')(x)
        x = Dense(1024, use_bias=False, activation='relu')(x)
        
        main_output = Dense(nbands, use_bias=True, activation=activation, name='main_output')(x)
        model = Model(inputs=[profile_input, surf_input], outputs=[main_output])

    else:
        main_output = Dense(nbands, activation=activation, name='main_output')(out) # use_bias=True, 
        model = Model(inputs=[profile_input], outputs=[main_output])
        
    return model


def train_model(model, data_splits, hyperparameter_dict, log_dir):
    """
    This function trains the input model with the given data samples in data_splits. 
    Hyperparameters use when fitting the model are defined in hyperparameter_dict.
    """
    tf_callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameter_dict['learning_rate'])
    model.compile(loss=hyperparameter_dict['loss'], optimizer=optimizer, metrics=['accuracy'])
    
    class_weights = None
    if 'class_weights' in hyperparameter_dict:
        class_weights = hyperparameter_dict['class_weights']
        
    history = model.fit(data_splits['X_train'], 
                        data_splits['y_train'], 
                        epochs=hyperparameter_dict['epochs'], 
                        batch_size=hyperparameter_dict['batch_size'], 
                        validation_data=(data_splits['X_val'], data_splits['y_val']), 
                        class_weight=class_weights,
                        callbacks=tf_callbacks,
                        verbose=True)
    return model, history


def log_history(history):
    prd_run = azureml.core.Run.get_context()
    # for k1, v1 in model.history.history.items():
    for k1, v1 in history.history.items():
        prd_run.log(k1, v1[-1])
    

def load_data_local(dataset_dir):
    print('loading all event data')
    dataset_dir = pathlib.Path(dataset_dir)
    prd_path_list = [p1 for p1 in dataset_dir.rglob(f'{MERGED_PREFIX}*{CSV_FILE_SUFFIX}') ]
    merged_df = pd.concat([pd.read_csv(p1) for p1 in prd_path_list])
    return merged_df


def log_history(history):
    prd_run = azureml.core.Run.get_context()
    # for k1, v1 in model.history.history.items():
    for k1, v1 in history.history.items():
        prd_run.log(k1, v1[-1])


def load_data_local(dataset_dir):
    print('loading all event data')
    dataset_dir = pathlib.Path(dataset_dir)
    prd_path_list = [p1 for p1 in dataset_dir.rglob(f'{MERGED_PREFIX}*{CSV_FILE_SUFFIX}') ]
    merged_df = pd.concat([pd.read_csv(p1) for p1 in prd_path_list])
    return merged_df


if USING_AZML:

    def load_data_azml_dataset(dataset_name):
        """
        This function loads data from AzureML storage and returns it as a pandas dataframe 
        """
        prd_run = azureml.core.Run.get_context()

        # We dont access a workspace in the same way in a script compared to a notebook, as described in the stackoverflow post:
        # https://stackoverflow.com/questions/68778097/get-current-workspace-from-inside-a-azureml-pipeline-step 
        prd_ws = prd_run.experiment.workspace

        dataset = azureml.core.Dataset.get_by_name(prd_ws, name=dataset_name)

        with dataset.mount() as ds_mount:
            print('loading all event data from azml file dataset')
            prd_path_list = [p1 for p1 in pathlib.Path(ds_mount.mount_point).rglob(f'{MERGED_PREFIX}*{CSV_FILE_SUFFIX}') ]
            merged_df = pd.concat([pd.read_csv(p1) for p1 in prd_path_list])
        return merged_df
    

def load_data_azure_blob(az_blob_cred, blob_path):
    """
    Load data direct from a blob store. Need to provide credentials
    Inputs
    az_blob_cred - A dictionary loaded from the credentials.json file.
    blob_path - The relative path to the object(s) in the blob store relative to the container specified in the credentials.
    """
    print('loading data direct from blobstore')
    container = az_blob_cred['container']
    acc_name = az_blob_cred['storage_acc_name']
    acc_key = az_blob_cred['storage_acc_key']

    prd_data_url = f'abfs://{container}/{blob_path}'

    handle1 = fsspec.open_files(
        prd_data_url,
        account_name=acc_name, 
        account_key=acc_key
    )

    fsspec_handle = fsspec.open(
        prd_data_url,
        account_name=acc_name, 
        account_key=acc_key
    )

    with fsspec_handle.open() as prd_data_handle:
        prd_merged_data = pd.read_csv(prd_data_handle)

    csv_list = []
    for h1 in list(handle1)[:2]:
        with h1.open() as prd_dh:
            csv_list += [pd.read_csv(prd_dh)]

    prd_merged_df = pd.concat(csv_list)
    return prd_merged_df

# create a load data function that takes a path on disk through the DatasetConfigConsumption object created by calling
# mydataset.as_download
# wheich get passed as an argument to the script
# argument will then contain the path to the folder with the files
    

def random_sample(df, test_fraction, random_state):
    """
    Sample test and train datasets randomly 
    Note: due to ensemble members being in different samples this 
    could leave to data leakage between train and test
    """
    n_samples = df.shape[0]
    test_df = df.sample(
        int(n_samples*test_fraction), random_state=random_state)
    train_df = df[~np.isin(df.index, test_df.index)]
    return train_df, test_df


def random_time_space_sample(df, test_fraction, random_state, sampling_columns):
    """
    Sample test and train dataset randomly over time and space,
    but keeps all ensemble members in the same sample
    """
    unique_time_location = df[df['realization']==0].groupby(sampling_columns)
    samples = unique_time_location.count().sample(frac=test_fraction, random_state=random_state).reset_index()[sampling_columns]

    samples_labelled = pd.merge(df, samples, how='left', left_on=sampling_columns, right_on=sampling_columns, indicator=True)

    test_df = samples_labelled[samples_labelled._merge=='both'].drop(columns='_merge', axis=1)
    train_df = samples_labelled[samples_labelled._merge=='left_only'].drop(columns='_merge', axis=1)
    
    return train_df, test_df


def sample_data_by_time(df, test_fraction=0.2, test_save=None, random_state=None):
    """
    Sample test and train datasets by selecting the last 20% of timesteps in the data for testing
    """
    n_timesteps = df.time.unique().size
    test_samples = np.round(n_timesteps / (1 / test_fraction)).astype('int')
    test_mask = np.isin(df['time'], df['time'].unique()[-test_samples:])
    test_df = df[test_mask]
    train_df = df[~test_mask]
    
    if test_save:
        container = test_save['datastore_credentials']['container']
        acc_name = test_save['datastore_credentials']['storage_acc_name']
        acc_key = test_save['datastore_credentials']['storage_acc_key']
        # # save test dataset
        # fsspec_handle = fsspec.open(
        #     f'abfs://{container}/{test_save["filename"]}_test.csv', account_name=acc_name, account_key=acc_key, mode='wt')
        # with fsspec_handle.open() as testfn:
        #     test_df.to_csv(testfn)
        # # save train dataset
        # fsspec_handle = fsspec.open(f'abfs://{container}/{test_save["filename"]}_train.csv', account_name=acc_name, account_key=acc_key, mode='wt')
        # with fsspec_handle.open() as trainfn:
        #     train_df.to_csv(trainfn)
    else: 
        return train_df, test_df


def reshape_profile_features(df, features, data_dims_dict):
    """
    This function reshapes vertical profile data from tabular data 
    back into data with a height dimension
    """
    prof_feature_columns = get_profile_columns(features, df.columns)
    df = df[prof_feature_columns]
    df = np.transpose(
        df.to_numpy().reshape(
            df.shape[0], 
            data_dims_dict['nprof_features'], 
            data_dims_dict['nheights']
            ),
        (0, 2, 1))
    return df


def get_profile_columns(profile_vars, columns):
    return [s for s in columns for vars in profile_vars if s.startswith(vars)]


def load_test_data(test_fn, feature_dict, data_dims_dict):
    """
    This function can be used to load in test data and returns 
    input and target data ready to be used to test an ML model.
    """
    test_data = pd.read_csv(test_fn)
    with open('standardScaler.pkl', 'rb') as fin:
        standardScaler = pickle.load(fin)

    prof_feature_columns = get_profile_columns(feature_dict['profile'], test_data.columns)
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


def create_test_train_datasets(data):
    with open('credentials_file.json') as cred:
        cred = json.load(cred)
    test_save = {
        'datastore_credentials' : cred, 
        'filename': 'storm_dennis', 
    }
    
    # Sample test and train datasets and save to blob storage
    sample_data(data, test_save=test_save, test_fraction=test_fraction, random_state=random_state)


def preprocess_data(input_data, feature_dict, test_fraction=0.2):
    """
    Preprocesses data ahead of ML modelling
    """

    # drop NaN values in the dataset

    data = input_data.dropna()
    data = data[data['radar_mean_rain_instant'] < 50]  # remove any spuriously high radar data point
    
    print(f"target has dims: {len(feature_dict['target'])}")
    if isinstance(feature_dict['target'], list):
        print(f"dropping smallest bin: {feature_dict['target'][0]}")
        # If feature_dict['target'] is length greater than 1, then the target 
        # is a set of intensity bands and so we drop data where the
        # smallest intensity band has a fraction of 1 
        # i.e. all radar cells in the model cell are in the lowest intensity band
        data = data[data[feature_dict['target'][0]]!=1]
    else:
        print(f'dropping zeros')
        # If feature_dict['target'] is length 1, then either mean or max precip is the target and so 
        # we drop data points with zero precip in the radar data
        data = data[data[feature_dict['target']]>0]

    # Get a list of columns names for profile features
    print('getting profile columns')
    prof_feature_columns = get_profile_columns(feature_dict['profile'], data.columns)

    data_dims_dict = {
        'nprof_features' : len(feature_dict['profile']),
        'nheights' : len(prof_feature_columns)//len(feature_dict['profile']),
        'nsinglvl_features' :len(feature_dict['single_level']),
    }
    
    if isinstance(feature_dict['target'], list):
        data_dims_dict['nbands'] = len(feature_dict['target'])
    else:
        data_dims_dict['nbands'] = 1
    
    print(data_dims_dict)
    random_state = np.random.RandomState() 
    # mlflow.log_metric('random state', random_state)

    # Extract and return train and validate datasets
    train_df, val_df = random_time_space_sample(
        data,
        test_fraction=test_fraction,
        random_state=random_state,
        sampling_columns=['time', 'latitude', 'longitude']
    )
    
    X_train = train_df[prof_feature_columns + feature_dict['single_level']]
    y_train = train_df[feature_dict['target']]
    X_val = val_df[prof_feature_columns + feature_dict['single_level']]
    y_val = val_df[feature_dict['target']]
    
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame()
    
    if isinstance(y_val, pd.Series):
        y_val = y_val.to_frame()
    
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

    #TODO: we should rather not save pickle if possible, theres no guarentee of being able to 
    # load in future. Would be better to save parameters as a JSON. We could probably save as 
    # a metric with the run so we can load in the same parameters when loading a saved model
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


def calc_metrics(data_splits, y_pred):
    metrics_dict = {}
    metrics_dict['mean_absolute_error'] = mean_absolute_error(data_splits['y_val'], y_pred)
    metrics_dict['R-squared score'] = r2_score(data_splits['y_val'], y_pred)

    for k1, v1 in metrics_dict.items():
        mlflow.log_metric(k1, v1)
    return metrics_dict


def save_model(model, prd_model_name):
    mlflow.keras.log_model(model, prd_model_name)


if USING_AZML:

    def calc_metrics_azml(data_splits, y_pred):
        metrics_dict = {}
        metrics_dict['mean_absolute_error'] = mean_absolute_error(data_splits['y_val'], y_pred)
        metrics_dict['R-squared score'] = r2_score(data_splits['y_val'], y_pred)
        current_run = azureml.core.Run.get_context()

        for k1, v1 in metrics_dict.items():
            current_run.log(k1, v1)
        return metrics_dict


    def save_model_azml(model, prd_model_name):
        prd_run = azureml.core.Run.get_context()

        # We dont access a workspace in the same way in a script compared to a notebook, as described in the stackoverflow post:
        # https://stackoverflow.com/questions/68778097/get-current-workspace-from-inside-a-azureml-pipeline-step 
        prd_ws = prd_run.experiment.workspace

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
