import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xarray as xr
# import datetime as dt
import cartopy.crs as ccrs

def compare_boxplots(data_all, data_w_precip, data_no_precip, title=None, ylabel=None, xlabel=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, figsize=(15,8))
    # All data
    data_all.plot.box(vert=False,ax=ax1)
    ax1.set_title('All data')
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)
    
    # With precip
    data_w_precip.plot.box(vert=False,ax=ax2)
    ax2.set_yticks([])
    ax2.set_title('Only data with radar precip > 0')
    ax2.set_xlabel(xlabel)
    
    # No precip
    data_no_precip.plot.box(vert=False,ax=ax3)
    ax3.set_yticks([])
    ax3.set_title('Only data where radar precip is 0')
    ax3.set_xlabel(xlabel)
    
    fig.suptitle(title)
    plt.show()


def plot_boxplot(data, title=None, ylabel=None):
    data.plot.box(rot=90, figsize=(10,4))
    plt.xlabel('Model level height')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_violinplot(data, title=None, ylabel=None):
    fig, axes = plt.subplots()
    sns.violinplot(data=data, ax=axes, scale='width')
    fig.set_size_inches(10, 4)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()


def plot_corr_heatmap(data):
    data_corr = data.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(data_corr, annot=True)
    plt.show()


def visualise_ens_spread(df, var, spread_var='std'):
    
    var_columns = [s for s in df.columns if s.startswith(var) and s.endswith('.0')]
    grouped_data = df.groupby(['latitude', 'longitude', 'forecast_reference_time'])[var_columns].agg(['std', 'min', 'max'])
    
    i, j = 0, 0
    fig, ax = plt.subplots(3, 11, figsize=(22,6), sharex=True, sharey=True)
    for col in var_columns:
        height = col.split('_')[-1]
        if spread_var == 'std':
            ax[i, j].hist(
                grouped_data[col]['std'], 
                bins=50)
            fig.suptitle(f'{var}')
            fig.supxlabel('Standard deviation')
        if spread_var == 'range':
            ax[i, j].hist(
                grouped_data[col]['max'] - grouped_data[col]['min'], 
                bins=30)
            fig.suptitle(f'Range - {var}')
            fig.supxlabel('Range')
        ax[i,j].set_title(f'{height}m')

        j += 1
        if j == 11:
            i += 1
            j = 0
    fig.text(0.1, 0.5, 'Frequency', va='center', ha='center', rotation='vertical')
    plt.show()
    
    
def ens_variability_map(df, var, height_level, time_index):
    var_columns = [s for s in df.columns if s.startswith(var) and s.endswith('.0')]
    grouped_data = df.groupby(['latitude', 'longitude', 'forecast_reference_time'])[var_columns].agg(['std', 'min', 'max'])

    height_col_idx = [col.split('_')[-1]==str(float(height_level)) for col in var_columns]
    height_column = np.array(var_columns)[height_col_idx][0]

    fig, ax = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12,5))
    
    data_to_plot = grouped_data[height_column]['std'].to_xarray().isel(forecast_reference_time=time_index)
    data_to_plot.plot.pcolormesh(ax=ax[0], transform=ccrs.PlateCarree())
    ax[0].coastlines()
    ax[0].set_title(f'{height_column} - ensemble standard deviation ')
    
    min_vars = grouped_data[height_column]['min'].to_xarray().isel(forecast_reference_time=time_index)
    max_vars = grouped_data[height_column]['max'].to_xarray().isel(forecast_reference_time=time_index)
    data_to_plot = max_vars - min_vars
    data_to_plot.plot.pcolormesh(ax=ax[1], transform=ccrs.PlateCarree())
    ax[1].coastlines()
    ax[1].set_title(f'{height_column} - ensemble range')
    
    plt.suptitle(f'{data_to_plot.forecast_reference_time.values}')
    
    plt.show()
    

def plot_avg_diff(df, radar_var, nwp_var, time_index):
    data_xr = df.set_index(['latitude', 'longitude', 'forecast_reference_time', 'realization']).to_xarray()
    
    # reduced realization dimension by taking the mean and time by selecting an index
    radar_data = data_xr[radar_var].mean(dim='realization').isel(forecast_reference_time=time_index)
    if 'max_rain' in radar_var :
        nwp_data = data_xr[nwp_var].max(dim='realization').isel(forecast_reference_time=time_index)
    else:
        nwp_data = data_xr[nwp_var].mean(dim='realization').isel(forecast_reference_time=time_index)
    
    # calculate vmax
    vmax = np.ceil(max(radar_data.max(), nwp_data.max()))
    
    # plot with three subplots
    # the first two panels shows radar and nwp data and final panel shows the difference
    fig, ax = plt.subplots(1, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15,5))
    
    radar_data.plot.pcolormesh(ax=ax[0], transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)
    ax[0].coastlines()
    ax[0].set_title('Radar precip (mm)')

    nwp_data.plot.pcolormesh(ax=ax[1], transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)
    ax[1].coastlines()
    ax[1].set_title('NWP ensemble precip (mm)')

    diff = nwp_data - radar_data
    diff.plot.pcolormesh(ax=ax[2], transform=ccrs.PlateCarree())
    ax[2].coastlines()
    ax[2].set_title('Diff between radar and NWP (mm)')
    
    plt.suptitle(f'{radar_data.forecast_reference_time.values}')

    plt.show()


def radar_fraction_bins_plot(df, target_var, examples, bins_dict, precip_type):
    assert len(examples)>=2, 'AssertionError please provide 2 or more example indices'
    data_target = df[target_var]
        
    colnames = [colname.split('_')[-1] for colname in data_target.columns]
    limits = np.array(colnames).astype(float)

    fig, ax = plt.subplots(len(examples), 1, figsize=(10,2.5*len(examples)), sharey=True, sharex=True)
    for i, example in enumerate(examples):

        # select MOGREPS-G members for location and time
        nwp_ens_data = df[
            (df['latitude']==df.loc[example, 'latitude']) & 
            (df['longitude']==df.loc[example, 'longitude']) & 
            (df['forecast_reference_time']==df.loc[example, 'forecast_reference_time'])
        ]['thickness_of_rainfall_amount']

        # calculate band probabilities from ensemble members
        num_in_bands = []
        for imp_ix, (imp_key, imp_bounds) in enumerate(bins_dict.items()):
            num_in_band = np.count_nonzero(
                (nwp_ens_data >=  imp_bounds[0]) & (nwp_ens_data <= imp_bounds[1]))
            num_in_bands.append(num_in_band / len(nwp_ens_data))

        # plot bar charts 
        ax[i].bar(colnames, data_target.loc[example], label='Radar')
        ax[i].bar(colnames, num_in_bands, fill=False, label='MOGREPS-G')
        ax[i].set_title(
            (f"time: {df.loc[example]['forecast_reference_time']} "
             f"location: ({df.loc[example]['latitude']}, {df.loc[example]['longitude']})"))

        # plot a line of max and mean for radar
        mean_rain = df.loc[example][f'radar_mean_rain_{precip_type}']
        ax[i].axvline(
            colnames[np.where(limits <= mean_rain)[0][-1]], 
            alpha=0.7, c='cyan', lw=2,
            label=f'mean radar precip ({mean_rain:.2f}mm)')

        max_rain = df.loc[example][f'radar_max_rain_{precip_type}']
        ax[i].axvline(
            colnames[np.where(limits <= max_rain)[0][-1]],
            c='green', lw=2,
            label=f'max radar_precip ({max_rain:.2f}mm)')

        # plot a line of max and mean for MOGREPS-G
        nwp_mean_rain = np.mean(nwp_ens_data)
        ax[i].axvline(
            colnames[np.where(limits <= nwp_mean_rain)[0][-1]],
            c='orange', ls='dotted',  
            label=f'mean nwp precip ({nwp_mean_rain:.2f}mm)')

        nwp_max_rain = np.max(nwp_ens_data)
        ax[i].axvline(
            colnames[np.where(limits <= nwp_max_rain)[0][-1]],
            c='darkorange', ls='dashed', 
            label=f'max nwp precip ({nwp_max_rain:.2f}mm)')

        ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
    ax[-1].set_xlabel('Precipitation band (mm)')  

    plt.show()
    

def binned_height_profiles(df, target_var, columns, y_label):
    labels = ['0% to 4%'] + [f'{i+1}% to {i+4}%' for i in range(4, 100, 4)]
    df['Bin'] = pd.qcut(df[target_var], np.arange(0,1.01, 0.04), labels=labels)
    grouped = df.groupby('Bin').agg('mean')

    # Plot the cloud fraction over height level
    # Sorted by 3hrly precip accumulations to assessed with the gradient varies significantly
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(21, 7), gridspec_kw={'height_ratios': [3, 1]})
    
    var_min = min([min(df[column]) for column in columns])
    var_max = max([max(df[column]) for column in columns])
    
    im1 = sns.heatmap(
        grouped[columns+[target_var]].sort_values(target_var).T[::-1],
        xticklabels=False, yticklabels=False, cmap='Blues', vmin=var_min, vmax=var_max, ax=ax1, cbar=False)
    ax1.set_ylabel(y_label)
    ax1.set_xticks(ax1.get_xticklabels(), ha="center")
    ax1.set_xlabel('')

    im2 = df[[target_var, 'Bin']].groupby('Bin').boxplot(
        subplots=False,
        column=[target_var],
        ax=ax2)

    ax2.set_xticklabels(ax2.get_xticklabels(),ha="center", rotation=90)
    ax2.set_ylabel('3hr precip acc (mm)')

    plt.show()