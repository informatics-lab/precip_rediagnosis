import pandas as pd
import numpy as np

def sample_data(df, test_fraction=0.2, savefn=None, random_state=None):
    n_samples = df.shape[0]
    test_dataset = df.sample(int(n_samples*test_fraction), random_state=random_state)
    train_dataset = df[~np.isin(df.index, test_dataset.index)]
    if savefn:
        test_dataset.to_csv(savefn)
        return train_dataset
    else: 
        return train_dataset, test_dataset

def seperate_target_feature(df, target, features, profiles=False):
    if profiles:
        prof_feature_columns = [s for s in df.columns for vars in features if s.startswith(vars)]
        features_df = df[prof_feature_columns]
        features_df = np.transpose(features_df.to_numpy().reshape(features_df.shape[0], len(features), len(prof_feature_columns)//len(features)), (0, 2, 1))
    else:
        features_df = df[features]
        
    target_df = df[[target]]
    
    return features_df, target_df
