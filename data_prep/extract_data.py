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

    parser1.add_argument('--output-level',
                         dest='output_level',
                         help='Level of output in the logs.',
                         default='info',
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
    event_name = dataset_config['event_name']

    logger1 = drivers.get_logger(cmd_args.log_dir,
                                 drivers.MassExtractor.LOGGER_KEY + '_' + event_name,
                                 cmd_args.output_level)

    logger1.info('Running extract and prepare workflow for precip rediagnosis.')
    logger1.info(f'reading config from {cmd_args.config_file}')

    start_dt = datetime.datetime.strptime(dataset_config['event_start'],
                                          DATETIME_PARSER)

    end_dt = datetime.datetime.strptime(dataset_config['event_end'],
                                        DATETIME_PARSER)
    dest_dir = pathlib.Path(cmd_args.output_path) / event_name
    if not dest_dir.is_dir():
        dest_dir.mkdir()

    logger1.info(f'processing config for event {event_name}')
    logger1.info(f'writing output to {dest_dir}')

    driver_init_args = {
        'opts': {},
        'dest': str(dest_dir),
        'date_range': [start_dt, end_dt],
        'log_dir': cmd_args.log_dir,
        'target_cube_path': cmd_args.target_cube_path,
        'target_time_delta': float(dataset_config['target_time_delta']),
        'date_fname_template': dataset_config['date_fname_template'],
        'fname_extension_grid': dataset_config['fname_extension_grid'],
        'fname_extension_tabular': dataset_config['fname_extension_tabular'],
        'output_level': cmd_args.output_level,
    }
    driver_list = []

    try:
        for data_source_cfg in dataset_config['data_sources']:
            logger1.info(f'processing data source of type {data_source_cfg["data_type"]}')
            driver_init_args['opts'] = data_source_cfg
            driver1 = drivers.extractor_factory(data_source_cfg['data_extractor'],
                                                driver_init_args)
            logger1.info(f'Running extract for {data_source_cfg["data_type"]}')
            logger1.info(f'Running extract for {data_source_cfg["data_type"]}')
            driver1.extract()
            logger1.info(f'Running prepare for {data_source_cfg["data_type"]}')
            driver1.prepare()
            driver_list += [driver1]

        logger1.info('merging data from different sources into one dataframe')
        merged_df = drivers.merge_prepared_output(
            extractor_list=driver_list,
            merge_vars=dataset_config['merge_vars'],
            merge_method='inner')

        # generate output filename and path
        start_dt = min(merged_df['time'])
        end_dt = max(merged_df['time'])
        fname_timestamp = dataset_config['date_fname_template'].format(start=start_dt, end=end_dt)
        merged_fname = dataset_config['merged_outpout_prefix'] + '_' + fname_timestamp + dataset_config['fname_extension_tabular']
        merged_output_path = dest_dir / merged_fname
        logger1.info(f'writing merged dataframe to {merged_output_path}')
        merged_df.to_csv(merged_output_path, index=False)
    except BaseException as e1:
        logger1.error(str(e1))
        raise e1

    logger1.info('processing completed successfully.')


if __name__ == '__main__':
    main()