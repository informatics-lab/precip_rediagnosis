#!/usr/bin/env python
'''
This script extracts MOGREPS-G from the Met Office MASS data archive for particular period. It extracts data for particular variables and stores them in the specified location. Aggregation of data and other processing is done separately.

Future work is to make this more general, and specify paramters through a JSON file so this can easily run for many different time periods, possibly as the backend to a data catalog of some sort.
'''
import pathlib
import datetime
import subprocess
import sys
import logging
root_mass =  'moose:/opfc/atm/mogreps-g/lev1/'

num_periods = 10
start_ref_time = datetime.datetime(2020,2,14,12)
forecast_ref_time_range = [start_ref_time + datetime.timedelta(hours=6)*i1 for i1 in range(num_periods)]
leadtime_hours = 6

variables_to_extract = [
    "cloud_amount_of_total_cloud",
    "cloud_amount_on_height_levels",
    "pressure_on_height_levels",
    "temperature_on_height_levels",
    "relative_humidity_on_height_levels",
    "wind_direction_on_height_levels",
    "wind_speed_on_height_levels",
    "rainfall_accumulation-PT03H",
    "snowfall_accumulation-PT03H",
    "rainfall_rate",
    "snowfall_rate",
    "height_of_orography",
    "pressure_at_mean_sea_level",
    'rainfall_rate',
    'rainfall_rate_from_convection',
    'rainfall_accumulation-PT03H',
    'rainfall_accumulation_from_convection-PT03H',
    'snowfall_rate',
    'snowfall_rate_from_convection',
    'snowfall_accumulation-PT03H',
    'snowfall_accumulation_from_convection-PT03H',
    'rainfall_rate_max-PT01H',
    'rainfall_rate_max-PT03H',
    'rainfall_rate_from_convection_max-PT03H',
    'snowfall_rate_max-PT01H',
    'snowfall_rate_max-PT03H',
    'snowfall_rate_from_convection_max-PT03H',
]

current_time = datetime.datetime.now()
logs_directory = pathlib.Path('/data/users/shaddad/precip_rediagnosis/logs')
current_timestamp = '{ct.year:04d}{ct.month:02d}{ct.day:02d}{ct.hour:02d}{ct.minute:02d}{ct.second:02d}'.format(ct=current_time)
logger = logging.getLogger('extract_mass')
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                              '%m-%d-%Y %H:%M:%S')

handler1 = logging.FileHandler(logs_directory / f'extract_mass_{current_timestamp}.log')
handler1.setLevel(logging.INFO)
handler1.setFormatter(formatter)
logger.addHandler(handler1)

handler1 = logging.StreamHandler(sys.stdout)
handler1.setLevel(logging.INFO)
handler1.setFormatter(formatter)
logger.addHandler(handler1)

logger.debug('Extracting files from mass')
logger.info(f'forecast reference times {forecast_ref_time_range}')
logger.info(f'variables to extract {variables_to_extract}')

mass_root = pathlib.Path('moose:/opfc/atm/')
dataset = 'mogreps-g'
subset = 'lev1'
forecast_ref_template = '{frt.year:04d}{frt.month:02d}{frt.day:02d}T{frt.hour:02d}00Z.nc.file'
fname_template = '{vt.year:04d}{vt.month:02d}{vt.day:02d}T{vt.hour:02d}00Z-PT{lead_time:04d}H00M-{var_name}.nc'
dest_root = pathlib.Path('/scratch/shaddad/precip_rediagnosis/storm_dennis_9hr_lt')
mass_cmd_template = 'moo get {args} {files} {dest_dir}'
output_dir = dest_root / dataset
if not output_dir.is_dir():
    logger.info('creating output directory {output_dir}')
    output_dir.mkdir()

for var1 in variables_to_extract:
    extract_path_list = []
    for fcst_ref_time in forecast_ref_time_range:
        validity_time = fcst_ref_time + datetime.timedelta(hours=leadtime_hours)
        mass_path = (mass_root /
                     dataset /
                     subset /
                     forecast_ref_template.format(frt=fcst_ref_time)
                     / fname_template.format(vt=validity_time,
                                             lead_time=leadtime_hours,
                                             var_name=var1)
                     )
        extract_path_list += [str(mass_path)]

    mass_get_cmd = mass_cmd_template.format(files=' '.join(extract_path_list),
                                            dest_dir=str(output_dir),
                                            args='-f')
    logger.info(f'running command:\n{mass_get_cmd}')

    try:
        cmd_output = subprocess.check_output(mass_get_cmd, shell=True)
    except subprocess.CalledProcessError as err1:
        logger.error(f'return code = {err1.returncode}\n'
                     f'output = {err1.output}\n'
                     f'error output = {err1.stderr}')

    logger.info(f'get command output:\n{cmd_output}')

logger.info(f'files output to {dest_root}')


