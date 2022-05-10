import datetime
import pathlib
import logging
import sys
import subprocess

import pandas

import iris
import iris.cube

def calc_dates_list(start_datetime, end_datetime, delta_hours):
    dates_to_extract = list(pandas.date_range(
        start=start_datetime,
        end=end_datetime - datetime.timedelta(seconds=1),
        freq=datetime.timedelta(
            hours=delta_hours)).to_pydatetime())
    return dates_to_extract


def run_shell_cmd(shell_cmd, logger):
    try:
        logger.info(f'running cmd:\n{shell_cmd}')
        cmd_output = subprocess.check_output(shell_cmd, shell=True)
    except subprocess.CalledProcessError as err1:
        logger.error(
            'return code = {err1.returncode}\n'
            f'output = {err1.output}\n'
            f'error output = {err1.stderr}')

    logger.info(f'get command output:\n{cmd_output}')
    return cmd_output


def get_logger(log_dir, logger_key):
    current_time = datetime.datetime.now()
    logs_directory = pathlib.Path(log_dir)
    current_timestamp = '{ct.year:04d}{ct.month:02d}{ct.day:02d}{ct.hour:02d}{ct.minute:02d}{ct.second:02d}'.format(
        ct=current_time)
    logger = logging.getLogger('extract_mass')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
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

def delete_file_list(path_list):
    for p1 in path_list:
        p1.unlink()

class MassExtractor():
    MASS_CMD_TEMPLATE = 'moo get {args} {src_paths} {dest_path}'
    UNZIP_CMD_TEMPLATE = 'gunzip {root}/{files}'
    LOGGER_KEY= 'extract_mass'

    def __init__(self, opts, dest, date_range, log_dir='logs'):
        self._opts = opts
        self._dest_path = pathlib.Path(dest)
        self._log_dir = log_dir
        self._date_range = date_range

        self._merge_data = None

        self._create_logger()

    def _create_logger(self):
        self.logger = get_logger(self._log_dir, MassExtractor.LOGGER_KEY)

    def extract(self):
        raise NotImplementedError()

    def prepare(self):
        raise NotImplementedError()

    def get_merge_data(self):
        return self._merge_data



class ModelStageExtractor(MassExtractor):

    def __init__(self, opts, dest, date_range, log_dir='logs'):
        super().__init__(opts=opts,
                         dest=dest,
                         date_range=date_range,
                         log_dir=log_dir,
                         )

    def extract(self):
        variables_to_extract = self._opts['variables']
        leadtime_hours = self._opts['leadtime']
        mass_root = self._opts['source_root']
        forecast_ref_time_range = calc_dates_list(self._date_range[0],
                                                  self._date_range[1],
                                                  self._opts['time_delta'],
                                                  )
        for var1 in variables_to_extract:
            extract_path_list = []
            for fcst_ref_time in forecast_ref_time_range:
                validity_time = fcst_ref_time + datetime.timedelta(
                    hours=leadtime_hours)
                mass_path = (mass_root /
                             self._opts['dataset'] /
                             self._opts['subset'] /
                             self._opts['forecast_ref_template'].format(frt=fcst_ref_time)
                             / self._opts['fname_template'].format(vt=validity_time,
                                                     lead_time=leadtime_hours,
                                                     var_name=var1)
                             )
                extract_path_list += [str(mass_path)]
            output_dir = self._dest_path / self._opts['dataset']
            mass_get_cmd = MassExtractor.MASS_CMD_TEMPLATE.format(
                src_paths=' '.join(extract_path_list),
                dest_path=str(output_dir),
                args='-f')

            run_shell_cmd(mass_get_cmd, self.logger)

            self.logger.info(f'files output to {self._dest_path}')

    def prepare(self):
        pass


class RadarExtractor(MassExtractor):
    def __init__(self, opts, dest, date_range, log_dir='logs'):
        super().__init__(opts=opts,
                         dest=dest,
                         date_range=date_range,
                         log_dir=log_dir,
                         )

    def extract(self):
        mass_root = self._opts['source_root']
        calc_dates_list
        fname_mass_template = self._opts['fname_mass_template']

        dates_to_extract = calc_dates_list(self._date_range[0],
                                           self._date_range[1],
                                           self._opts['archive_time_chunk'],
                                           )
        fnames_to_extract = [fname_mass_template.format(dt=dt1)
                             for dt1 in dates_to_extract]

        paths_to_extract = [mass_root + mass_fname
                            for mass_fname in fnames_to_extract]

        # First run the MASS extract
        dest_root = self._dest_path / self._opts['dataset']

        if not dest_root.is_dir():
            dest_root.mkdir()

        mass_radar_get_cmd = MassExtractor.MASS_CMD_TEMPLATE.format(
            src_paths=' '.join(paths_to_extract),
            dest_path=str(dest_root),
            args='-f',
        )
        run_shell_cmd(mass_radar_get_cmd, self.logger)

        #Next untar the files that comes out of mass
        untar_cmd_template = 'tar -xf {path} --directory {dest_root}'
        tars_to_delete = []
        for fname1 in fnames_to_extract:
            extracted_path = dest_root / fname1
            tars_to_delete += [extracted_path]
            untar_cmd = untar_cmd_template.format(path=extracted_path,
                                                  dest_root=dest_root)
            cmd_output = run_shell_cmd(untar_cmd, self.logger)

        # Next unzip the files extracted from the tar ball that relate to the
        # products we are interested in. All the files that remained zipped
        # will be cleaned up (i.e. deleted) immediately after this step
        for p1 in self._opts['products'].values():
            unzip_cmd = MassExtractor.UNZIP_CMD_TEMPLATE.format(
                root=dest_root,
                files=self._opts['variable_fname_template'].format(
                    timestamp='*',
                                                     product=p1,
                                                     resolution=self._opts['resolution'],
                                                     area=self._opts['area'],
                                                     ) + '.gz')
            cmd_output = run_shell_cmd(unzip_cmd, self.logger)

        self.logger.info('deleting tar files')
        delete_file_list(tars_to_delete)

        # delete the zip files from products we are not using
        self.logger.info('deleting unused gz files')
        delete_file_list(
            [p1 for p1 in dest_root.iterdir() if '.gz' in str(p1)])

        # finally combine the files into per day data
        for pname1, p1 in self._opts['products'].items():
            for selected_day in dates_to_extract:
                self.logger.info(f'extracting product {pname1} for {selected_day}')
                start_time_day = selected_day
                end_time_day = selected_day + datetime.timedelta(days=1)
                file_times = pandas.date_range(
                    start=start_time_day,
                    end=end_time_day - datetime.timedelta(seconds=1),
                    freq=datetime.timedelta(
                        seconds=300)).to_pydatetime().tolist()
                radar_day_pathlist = [
                    dest_root / self._opts['variable_fname_template'].format(
                        timestamp=f'{dt1.year:04d}{dt1.month:02d}{dt1.day:02d}{dt1.hour:02d}{dt1.minute:02d}',
                        product=p1,
                        area=self._opts['area'],
                        resolution= self._opts['resolution'],
                    )
                    for dt1 in file_times]

                radar_day_cubelist = iris.cube.CubeList(
                    [iris.load_cube(str(p1)) for p1 in radar_day_pathlist])
                iris.util.equalise_attributes(radar_day_cubelist)
                radar_day_cube = radar_day_cubelist.merge_cube()
                output_path = (
                        dest_root /
                        f'{pname1}_{selected_day.year:04d}'
                        f'{selected_day.month:02d}{selected_day.day}.nc')
                iris.save(radar_day_cube, str(output_path))
                self.logger.info(f'day output to {output_path}')
                delete_file_list(radar_day_pathlist)
        self.logger.info(f'files output to {dest_root}')

    def prepare(self):
        pass


class DummyExtractor(MassExtractor):

    def __init__(self, opts, dest, date_range, log_dir='logs'):
        super().__init__(opts=opts, dest=dest, date_range=date_range,
                         log_dir=log_dir)

    def extract(self):
        pass

    def prepare(self):
        pass


EXTRACTORS = {
    'ModelStageExtractor': ModelStageExtractor,
    'RadarExtractor': RadarExtractor,
}

def extractor_factory(extractor_str, init_args):
    """
    This is a factory method (based on the factory design pattern) to create a class derived from Mass Extractor, based on a key specified in the config JSON file.
    :param extractor_str: the key from the config file to select the relevant extractor class
    :param init_args: A dictionary of arguments to pass to the init function of the extractor class been constructed.
    :return: An object of the ssscted data extractor class.
    """
    driver_class =EXTRACTORS[extractor_str]
    driver_obj = driver_class(**init_args)
    return driver_obj

# create extra file to read ion args from command line and path to opts file
# write each lead time's data to a separate location
# port notebook to ascript to run after extractor driver, with a prepare function in the driver
# think about how the merging will work