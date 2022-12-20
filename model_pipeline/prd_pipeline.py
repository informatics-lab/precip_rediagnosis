# core python imports
import argparse
import tempfile
import pathlib
import datetime

# third party imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs

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
        
    history = model.fit(data_splits['X_train'], 
                        data_splits['y_train'], 
                        epochs=hyperparameter_dict['epochs'], 
                        batch_size=hyperparameter_dict['batch_size'], 
                        validation_data=(data_splits['X_val'], data_splits['y_val']), 
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


def calculate_permutation_feature_importance(model, data_splits, feature_dict, baseline_metric, npermutations):
    # permute by shuffling data
    feature_names = data_splits['profile_features_order'] + feature_dict['single_level']
    permutation_importance = {key:[] for key in feature_names}
    
    for ifeature, feature in enumerate(feature_names):
        print(f'permuting feature: {feature}')
        for iperm in np.arange(npermutations):
            if len(feature_dict['single_level']) > 0:
                X_val_permute = [data_splits['X_test'][0].copy(), data_splits['X_test'][1].copy()]
                if feature in feature_dict['single_level']:
                    X_val_permute[1][feature] = X_val_permute[1][feature].reindex(
                        np.random.permutation(X_val_permute[1][feature].index)).values
                else:
                    X_val_permute[0][..., ifeature] = np.take(
                        X_val_permute[0][..., ifeature],
                        indices=np.random.permutation(X_val_permute[0].shape[0]),
                        axis=0)

            else:
                X_val_permute = data_splits['X_test'].copy()
                X_val_permute[..., ifeature] = np.take(
                    X_val_permute[..., ifeature],
                    indices=np.random.permutation(X_val_permute.shape[0]),
                    axis=0)

            y_pred = model.predict(X_val_permute)

            permuted_metric = tf.keras.metrics.KLDivergence()
            permuted_metric.update_state(data_splits['y_test'], y_pred)
            permuted_metric = permuted_metric.result().numpy()

            permutation_importance[feature].append(permuted_metric - baseline_metric)
    return permutation_importance


def unify_dates(date_str):
    try:
        return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
    except: 
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d %H:%M:%S')


def preprocess_test_data(test_data, feature_dict, data_dims_dict):
    """
    This function can be used to load in test data and returns 
    input and target data ready to be used to test an ML model.
    """

    if isinstance(test_data, pathlib.PosixPath):
        test_data = pd.read_csv(test_data)
    
    test_data['time'] = test_data.time.apply(lambda x: unify_dates(x))

    with open('standardScaler.pkl', 'rb') as fin:
        standardScaler = pickle.load(fin)

    prof_feature_columns = get_profile_columns(feature_dict['profile'], test_data.columns)
    features = test_data[prof_feature_columns + feature_dict['single_level']]
    y_test = test_data[feature_dict['target']]

    X_test_scaled = pd.DataFrame(
        standardScaler.transform(features), 
        columns=features.columns,
        index=features.index)
    
    X_test = reshape_profile_features(
        X_test_scaled, feature_dict['profile'], data_dims_dict)
    if len(feature_dict['single_level']) > 0:
        X_test = [X_test, X_test_scaled[feature_dict['single_level']]]
    
    nwp_test = test_data[feature_dict['nwp']]
    meta_test = test_data[feature_dict['metadata']]
    
    data_splits = {
        'X_test': X_test,
        'y_test': y_test, 
        'nwp_test': nwp_test, 
        'meta': meta_test, 
        'profile_features_order': [*dict.fromkeys(['_'.join(column.split('_')[:-1]) for column in prof_feature_columns]).keys()]
    }
    
    return data_splits


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
        print(f"dropping datapoints where smallest bin {feature_dict['target'][0]} = 1")
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
    
    nwp_train = train_df[feature_dict['nwp']]
    nwp_val = val_df[feature_dict['nwp']]
    
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame()
    
    if isinstance(y_val, pd.Series):
        y_val = y_val.to_frame()
        
    if isinstance(nwp_train, pd.Series):
        nwp_train = nwp_train.to_frame()
        
    if isinstance(nwp_val, pd.Series):
        nwp_val = nwp_val.to_frame()
    
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

    data_splits = {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val' : y_val,
        'nwp_train' : nwp_train,
        'nwp_val' : nwp_val, 
        'meta_train': train_df[feature_dict['metadata']],
        'meta_val': val_df[feature_dict['metadata']],
        'profile_features_order': [*dict.fromkeys(['_'.join(column.split('_')[:-1]) for column in prof_feature_columns]).keys()],
    }

    return data_splits, data_dims_dict


def calculate_p_exceedance(df, data_source, bands, intensity_band_template):
    data_bands = [intensity_band_template.format(source=data_source, band_centre=threshold) for threshold in bands.keys()]
    data_exceedence_names = [band+'_exceedence' for band in data_bands]
    df[data_exceedence_names] = 1 - df[data_bands].cumsum(axis=1)
    return df


def fss(obs, fx):
    """ 
    The inputs to this function are the cumulative probability/fraction of exceeding a given precipitation threshold
    """
    FBS = ((fx - obs)**2).sum()
    FBS_ref = (fx**2).sum() + (obs**2).sum()
    FSS = 1 - (FBS/FBS_ref)
    return FSS


def freq_bias(obs, fx):
    """
    freq_bias = 0 indicates fx=0 and obs>0 
    0 < freq_bias < 1 indicates fx < obs
    very large frequency bias indicates fx >> obs (or very small obs value)
    freq_bias > 1 indicates fx > obs
    freq_bias = inf when obs=0
    """
    freq_bias = fx / obs
    freq_bias.replace([np.inf, -np.inf], np.nan, inplace=True)
    return freq_bias.mean(skipna=True)


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

            
# plotting function 
def calculate_metric(df, model, feature_dict, data_dims_dict, metric):
    data_splits = preprocess_test_data(df, feature_dict, data_dims_dict)     

    ypred_test_df = model.predict(data_splits['X_test'])
    
    if metric == 'fss':
        func = fss
        y_pred = 1 - ypred_test_df.cumsum(axis=1)
        y_pred[y_pred<0]=0  # Some cumulative fractions sum to just over 1 due to rounding error

        y_test = 1 - data_splits['y_test'].cumsum(axis=1)
        y_test[y_test<0]=0  # Some cumulative fractions sum to just over 1 due to rounding error

        nwp_test = 1 - data_splits['nwp_test'].cumsum(axis=1)
        nwp_test[nwp_test<0]=0  # Some cumulative fractions sum to just over 1 due to rounding error
            
    if metric == 'freq_bias':
        func = freq_bias
        y_pred = ypred_test_df
        y_test = data_splits['y_test']
        nwp_test = data_splits['nwp_test']
   
    # calculative fractional skill score 
    ml_metric, nwp_metric = [], [] 
    for i, col in enumerate(feature_dict['target']):       
        
        ml = y_pred[:,i]
        nwp = nwp_test.iloc[:,i]
        radar = y_test.iloc[:,i]
        
        # Currently not in use as datasets not large enough, 
        # but for FSS need to remove instances where forecast is zero
        #         radar = radar[ml > 0]
        #         nwp = nwp[ml > 0] 
        #         ml = ml[ml > 0]

        #         radar = radar[nwp > 0]
        #         ml = ml[nwp > 0]
        #         nwp = nwp[nwp > 0]
    
        ml_metric.append(func(radar, ml))
        nwp_metric.append(func(radar, nwp))
    
    ml_metric_names = ['_'.join([f'ml_{metric}'] + [name.split('_')[-1]]) for name in feature_dict['target']]
    nwp_metric_names = ['_'.join([f'nwp_{metric}'] + [name.split('_')[-1]]) for name in feature_dict['target']]
    
    return pd.concat([pd.Series(ml_metric, index=ml_metric_names), pd.Series(nwp_metric, index=nwp_metric_names)])


def plot_metric_on_map(xrds, threshold, metric):
    """
    Produces a 2D map visualisation with three panels
    The first panel shows radar fraction of precipitation 
    The second panel shows either ML model fraction prediction or NWP probabilities, depending on 
    whether fx_source is 'ml' or 'mogrepsg'
    The final panel shows the difference between radar and predicted precipitation 
    """
    
    ml_data = xrds[f'ml_{metric}_{threshold}']
    nwp_data = xrds[f'nwp_{metric}_{threshold}']
    
    vmin = min(ml_data.min(), nwp_data.min())
    vmax = max(ml_data.max(), nwp_data.max())
    
    norm = None
    if metric == 'freq_bias':
        if vmin == 0:
            vmin=0.00001
        norm = matplotlib.colors.CenteredNorm(vcenter=1.0)
        cmap = matplotlib.cm.RdBu_r
    elif metric == 'fss':
        vmin = max(vmin, 0)
        vmax = min(vmax, 1)
        cmap = matplotlib.cm.viridis
   
    # plot with three subplots
    # the first two panels shows radar and nwp data and final panel shows the difference
    fig, ax = plt.subplots(1, 3, subplot_kw={'projection': ccrs.Mercator()}, figsize=(15,5))
    
    extent= (-5.65, 1.7800, 49.9600, 55.65)
    ml_data.plot.pcolormesh(ax=ax[0], transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
    ax[0].set_extent(extent)
    ax[0].coastlines()

    nwp_data.plot.pcolormesh(ax=ax[1], transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
    ax[1].set_extent(extent)
    ax[1].coastlines()
    
    diff = ml_data - nwp_data
    diff.plot.pcolormesh(ax=ax[2], cmap=cmap, transform=ccrs.PlateCarree(), norm=matplotlib.colors.CenteredNorm(vcenter=0.0))
    ax[2].set_extent(extent)
    ax[2].coastlines()

    return fig, ax


def plot_forecast(xrds, threshold, fx_source, exceedance_val, time_idx):
    """
    Produces a 2D map visualisation with three panels
    The first panel shows radar fraction of precipitation 
    The second panel shows either ML model fraction prediction or NWP probabilities, depending on 
    whether fx_source is 'ml' or 'mogrepsg'
    The final panel shows the difference between radar and predicted precipitation 
    """
    
    fx_data = xrds[f'{fx_source}_fraction_in_band_instant_{threshold}_exceedence'].isel(time=time_idx)
    radar_data = xrds[f'radar_fraction_in_band_instant_{threshold}_exceedence'].isel(time=time_idx)
  
    # plot with three subplots
    # the first two panels shows radar and model prediction data and final panel shows the difference
    fig, ax = plt.subplots(1, 3, figsize=(15,5), subplot_kw={'projection': ccrs.Mercator()}, )

    extent= (-5.65, 1.7800, 49.9600, 55.65)
    radar_data.plot.pcolormesh(ax=ax[0],vmin=0, vmax=1, transform=ccrs.PlateCarree())
    ax[0].set_extent(extent)
    ax[0].coastlines()
    ax[0].set_title(f'Radar fraction of \n precip > {exceedance_val}mm')
    
    fx_data.plot.pcolormesh(ax=ax[1],vmin=0, vmax=1,  transform=ccrs.PlateCarree())
    ax[1].set_extent(extent)
    ax[1].coastlines()
    ax[1].set_title(f'{fx_source} model fraction of \n precip > {exceedance_val}mm')

    diff = fx_data - radar_data
    diff.plot.pcolormesh(ax=ax[2],vmin=-1, vmax=1, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax[2].set_extent(extent)
    ax[2].coastlines()
    ax[2].set_title(f'Different between \n forecast and radar')
    
    plt.suptitle(f'{radar_data.time.values}')
    plt.show()
    

def postage_stamp_plot(datadf, threshold, time_idx, bands):
    
    test_xr = datadf.set_index(['time', 'latitude', 'longitude', 'realization']).to_xarray()
    test_timeslice = test_xr.isel(time=time_idx).dropna(dim='realization', how='all')
    
    ml_column = f'ml_fraction_in_band_instant_{threshold}_exceedence'
    fig, ax = plt.subplots(3, 6, subplot_kw={'projection': ccrs.Mercator()}, figsize=(30,12))
    i, j = 0, 0
    for realization in test_timeslice.realization.values:
        test_timeslice.sel(realization=realization)[ml_column].plot.pcolormesh(ax=ax[i,j],vmin=0, vmax=1,  transform=ccrs.PlateCarree())
        extent= (-5.65, 1.7800, 49.9600, 55.65)
        ax[i,j].set_extent(extent)
        ax[i,j].coastlines()
        ax[i,j].set_title(f'member={realization}')
        cbar = ax[i,j].collections[0].colorbar
        cbar.set_label('')
        j += 1
        if j == 6:
            i += 1
            j = 0
    fig.suptitle(f'ML fraction of precip > {bands[threshold][1]}mm  \n {test_timeslice.time.values}')
    plt.show()
    

def metric_barchart(ml_metric, nwp_metric, class_names, plot_annotations):
    x = np.arange(len(class_names))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(7,5.5))
    ax.bar(x + width/2, ml_metric, width, label='ML')
    ax.bar(x - width/2, nwp_metric, width, label='NWP')
    
    rects = ax.patches
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height, height.round(2), ha="center", va="bottom"
        )
    # Add some text for labels, title and custom x-axis tick labels, etc.
    
    plt.xticks(np.arange(5), class_names, rotation=45)
    ax.legend()
    
    ax.set_ylabel(plot_annotations['ylabel'])
    ax.set_xlabel(plot_annotations['xlabel'])
    ax.set_title(plot_annotations['title'])

    fig.tight_layout()

    return fig, ax


def make_saliency_map(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds)
        
        class_channel = preds[:,pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0,1))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()