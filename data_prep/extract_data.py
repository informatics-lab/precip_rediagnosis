#core python inputs
import pathlib
import argparse
import json
import datetime

# thid-party libraries
import pandas

# import packages from this library
import pprint
import drivers

DATETIME_PARSER= '%Y-%m-%dT%H:%MZ'


def get_args():
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--output-path',
                         dest='output_path',
                         help='Path on disk where extracted and prepared data '
                              'will be written.',
                         )
    parser1.add_argument('--config-file',
                         dest='config_file',
                         help='Path to the JSON file containing the parameters '
                              'for the data extraction operation.'
                         )

    parser1.add_argument('--log-dir',
                         dest='log_dir',
                         help='Directory to write log files to.',
                         default='logs',
                         )

    parser1.add_argument('--target-cube-path',
                         dest='target_cube_path',
                         help='NetCDF file pointing to a cube containing a regridding target.',
                         default='target_cube.nc',
                         )

    args1 = parser1.parse_args()
    return args1

def merge_data(drivers_list, merge_vars, output_path):
    merged_data = pandas.merge([d1.get_merge_data() for d1 in drivers_list], on=merge_vars)
    merged_data.to_csv(output_path)

def main():
    cmd_args = get_args()

    with open(cmd_args.config_file) as config_file:
        dataset_config = json.load(config_file)

    start_dt = datetime.datetime.strptime(dataset_config['event_start'],
                                          DATETIME_PARSER)

    end_dt = datetime.datetime.strptime(dataset_config['event_end'],
                                        DATETIME_PARSER)

    driver_init_args = {
        'opts': {},
        'dest': cmd_args.output_path,
        'date_range': [start_dt, end_dt],
        'log_dir': cmd_args.log_dir,
        'target_cube_path': cmd_args.target_cube_path,
        'target_time_delta': float(dataset_config['target_time_delta']),
        'date_fname_template': dataset_config['date_fname_template'],
        'fname_extension_grid': dataset_config['fname_extension_grid'],
        'fname_extension_tabular': dataset_config['fname_extension_tabular'],
    }
    driver_list = []
    for data_source_cfg in dataset_config['data_sources']:
        driver_init_args['opts'] = data_source_cfg
        print('extracting data from source with config as follows:')
        pprint.pprint(data_source_cfg)
        driver1 = drivers.extractor_factory(data_source_cfg['data_extractor'],
                                            driver_init_args)
        driver1.extract()
        driver1.prepare()
        driver_list += [driver1]

    # merge_data(driver_list)


if __name__ == '__main__':
    main()