#!/usr/bin/env python
'''
This script extracts radar data from the Met Office MASS data archive for particular period. It then selects the particular radar products of interest and then aggregates the data into 1 day of data per file (for easy access without making files too large). 

Current time period - 14/02/2022 to 18/02/2022
Variables - Composite rainfall and composite quality.

Future work is to make this more general, and specify paramters through a JSON file so this can easily run for many different time periods, possibly as the backend to a data catalog of some sort.
'''
import pathlib
import datetime
import subprocess
import sys
import logging
import pdb

import pandas
import iris

def get_logger(logs_directory):
    current_time = datetime.datetime.now()
    current_timestamp = '{ct.year:04d}{ct.month:02d}{ct.day:02d}{ct.hour:02d}{ct.minute:02d}{ct.second:02d}'.format(
        ct=current_time)
    logger = logging.getLogger('extract_mass_radar')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                  '%m-%d-%Y %H:%M:%S')

    handler1 = logging.FileHandler(
        logs_directory / f'extract_mass_{current_timestamp}.log')
    handler1.setLevel(logging.INFO)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    handler1 = logging.StreamHandler(sys.stdout)
    handler1.setLevel(logging.INFO)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)
    return logger

def run_shell_cmd(shell_cmd, logger):
    try:
        logger.info(f'running cmd:\n{shell_cmd}')
        cmd_output = subprocess.check_output(shell_cmd, shell=True)
    except subprocess.CalledProcessError as err1:
        logger.error(
            f'return code = {err1.returncode}\noutput = {err1.output}')
    logger.info(f'get command output:\n{cmd_output}')
    return cmd_output

def delete_file_list(path_list):
    for p1 in path_list:
        p1.unlink()

logs_directory = pathlib.Path('/data/users/shaddad/precip_rediagnosis/logs')
logger = get_logger(logs_directory)

start_datetime = datetime.datetime(2020,2,14,0,0)
end_datetime = datetime.datetime(2020,2,19,0,0)

dates_to_extract = pandas.date_range(start=start_datetime,
                  end=end_datetime-datetime.timedelta(seconds=1),
                  freq=datetime.timedelta(hours=24)).to_pydatetime().tolist()

# produce list of mass files
mass_root = 'moose:/adhoc/projects/radar_archive/data/comp/products/composites/'
fname_mass_template = '{dt.year:04d}{dt.month:02d}{dt.day:02d}.tar'
fnames_to_extract = [fname_mass_template.format(dt=dt1)
                    for dt1 in dates_to_extract]

paths_to_extract = [mass_root + mass_fname
                    for mass_fname in fnames_to_extract]

dest_root = pathlib.Path('/scratch/shaddad/precip_rediagnosis/radar2')
mass_get_cmd_template = 'moo get {src_paths} {dest_path}'
mass_radar_get_cmd = mass_get_cmd_template.format(
    src_paths=' '.join(paths_to_extract),
    dest_path = str(dest_root)
)
run_shell_cmd(mass_radar_get_cmd, logger)

untar_cmd_template = 'tar -xf {path} --directory {dest_root}'
tars_to_delete = []
for fname1 in fnames_to_extract:
    extracted_path = dest_root / fname1
    tars_to_delete += [extracted_path]
    untar_cmd = untar_cmd_template.format(path=extracted_path,
                                          dest_root=dest_root)
    cmd_output = run_shell_cmd(untar_cmd, logger)

variable_names = ['']
resolution = '1km'
area = 'UK'
products = {
    'composite_rainfall': 'nimrod_ng_radar_rainrate_composite',
    'composite_quality': 'nimrod_ng_radar_qualityproduct_composite'
}

variable_fname_template = '{timestamp}_{product}_{resolution}_{area}'
for p1 in products.values():
    unzip_cmd = 'gunzip {root}/{files}'.format(
        root=dest_root,
        files=variable_fname_template.format(timestamp='*',
                                             product=p1,
                                             resolution=resolution,
                                             area=area,
                                             ) + '.gz')
    cmd_output = run_shell_cmd(unzip_cmd, logger)

# after extracting what we want, delete all the files with gz extension
# remaining
logger.info('deleting tar files')
delete_file_list(tars_to_delete)

logger.info('deleting unused gz files')
delete_file_list([p1 for p1 in dest_root.iterdir() if '.gz' in str(p1)])

# load data for each day into iris
for pname1, p1 in products.items():
    for selected_day in dates_to_extract:
        logger.info(f'extracting product {pname1} for {selected_day}')
        start_time_day = selected_day
        end_time_day = selected_day + datetime.timedelta(days=1)
        file_times = pandas.date_range(
            start=start_time_day,
            end=end_time_day - datetime.timedelta(seconds=1),
            freq=datetime.timedelta(seconds=300)).to_pydatetime().tolist()
        radar_day_pathlist = [
            dest_root / variable_fname_template.format(
                timestamp=f'{dt1.year:04d}{dt1.month:02d}{dt1.day:02d}{dt1.hour:02d}{dt1.minute:02d}',
                product=p1,
                area=area,
                resolution=resolution,
            )
            for dt1 in file_times]

        radar_day_cubelist = iris.cube.CubeList(
            [iris.load_cube(str(p1)) for p1 in radar_day_pathlist])
        iris.util.equalise_attributes(radar_day_cubelist)
        radar_day_cube = radar_day_cubelist.merge_cube()
        output_path = dest_root / f'{pname1}_{selected_day.year:04d}{selected_day.month:02d}{selected_day.day}.nc'
        iris.save(radar_day_cube, str(output_path))
        logger.info(f'day output to {output_path}')
        delete_file_list(radar_day_pathlist)

logger.info(f'files output to {dest_root}')


