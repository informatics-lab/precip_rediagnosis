import datetime
import pathlib
import logging
import sys
import subprocess
import functools

import numpy
import pandas
import xarray

import iris
import iris.cube
import iris.quickplot
import iris.coord_categorisation

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

def compare_time(t1, t2):
    is_match = (t1.year == t2.year) and  (t1.month == t2.month) and  (t1.day == t2.day) and  (t1.hour== t2.hour) and  (t1.minute == t2.minute)
    return is_match

def cftime_to_datetime(input_cft):
    return datetime.datetime(input_cft.year,
                             input_cft.month,
                             input_cft.day,
                             input_cft.hour,
                             input_cft.minute,
                             input_cft.second,
                            )

def load_ds(ds_path, selected_bounds):
    try:
        subset1 = dict(selected_bounds)
        subset1['bnds'] = 0
        single_level_ds = xarray.load_dataset(ds_path).sel(**subset1)
    except KeyError as e1:
        single_level_ds = None
    return single_level_ds

class MassExtractor():
    MASS_CMD_TEMPLATE = 'moo get {args} {src_paths} {dest_path}'
    UNZIP_CMD_TEMPLATE = 'gunzip {root}/{files}'
    LOGGER_KEY= 'extract_mass'

    def __init__(self, opts_dict):
        self._opts = opts_dict['opts']
        self._dest_path = pathlib.Path(opts_dict['dest'])
        try:
            self._log_dir = opts_dict['log_dir']
        except KeyError:
            self._log_dir = 'logs'
        self._date_range = opts_dict['date_range']
        self._target_cube_path = opts_dict['target_cube_path']
        self._target_time_delta = opts_dict['target_time_delta']
        self._date_fname_template = opts_dict['date_fname_template']
        self._fname_extension_grid = opts_dict['fname_extension_grid']
        self._fname_extension_tabular = opts_dict['fname_extension_tabular']

        self._merge_data = None

        self._create_logger()

        # This rperesents a list of validity times, i.e. the observation times
        # that we are interested in. When dealing with forecast data, the
        # forecast reference time will be these values (validity times), minus
        # the lead time for the particular forecast being processed.
        self._target_time_range = calc_dates_list(self._date_range[0],
                                                  self._date_range[1],
                                                  self._target_time_delta,
                                                  )

    def _create_logger(self):
        self.logger = get_logger(self._log_dir, MassExtractor.LOGGER_KEY)

    def extract(self):
        raise NotImplementedError()

    def prepare(self):
        raise NotImplementedError()

    @property
    def data_to_merge(self):
        return self._merge_data



class ModelStageExtractor(MassExtractor):

    def __init__(self, opts_dict):
        super().__init__(opts_dict=opts_dict)

    def extract(self):
        sl_vars = self._opts['single_level_variables']
        hl_vars = self._opts['height_level_variables']
        variables_to_extract = sl_vars + hl_vars
        leadtime_hours = self._opts['leadtime']
        mass_root = pathlib.Path(self._opts['source_root'])
        validity_times = calc_dates_list(self._date_range[0],
                                                  self._date_range[1],
                                                  self._opts['time_delta'],
                                                  )

        forecast_ref_time_range = [
            vt - datetime.timedelta(hours=leadtime_hours)
            for vt in validity_times]

        output_dir = self._dest_path / self._opts['dataset']
        if not output_dir.is_dir():
            output_dir.mkdir()
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


            mass_get_cmd = MassExtractor.MASS_CMD_TEMPLATE.format(
                src_paths=' '.join(extract_path_list),
                dest_path=str(output_dir),
                args='-f')

            run_shell_cmd(mass_get_cmd, self.logger)

            self.logger.info(f'files output to {self._dest_path}')

    def prepare(self):
        num_realisations = int(self._opts['number_of_realisations'])

        sl_vars = self._opts['single_level_variables']
        hl_vars = self._opts['height_level_variables']
        variables_to_extract = sl_vars + hl_vars

        output_dir = self._dest_path / self._opts['dataset']

        dataset = self._opts['dataset']
        subset = self._opts['subset']
        forecast_ref_template = '{frt.year:04d}{frt.month:02d}{frt.day:02d}T{frt.hour:02d}00Z.nc.file'
        fname_template = '{vt.year:04d}{vt.month:02d}{vt.day:02d}T{vt.hour:02d}00Z-PT{lead_time:04d}H00M-{var_name}.nc'

        leadtime_hours = self._opts['leadtime']

        # load a cube for each variable in iris to get the actual variable name, and populate dictionary mapping from the var name in the file name to the variable as loaded into iris/xarray
        file_to_var_mapping = {
            var_file_name: iris.load_cube(
                str(output_dir / fname_template.format(
                    vt=self._target_time_range[0],
                    lead_time=leadtime_hours,
                    var_name=var_file_name))).name()
            for var_file_name in variables_to_extract}
        heights = iris.load_cube(
            str(output_dir / fname_template.format(
                vt=self._target_time_range[0],
                lead_time=leadtime_hours,
                var_name=
                hl_vars[0]))).coord('height').points
        merge_coords = ['latitude', 'longitude', 'time', 'realization']
        single_level_var_mappings = {v1: file_to_var_mapping[v1] for v1 in
                                     sl_vars}
        height_level_var_mappings = {v1: file_to_var_mapping[v1] for v1 in
                                     hl_vars}

        target_grid_cube = iris.load_cube(
            str(self._target_cube_path)
        )


        uk_bounds = {
            'latitude': (min(target_grid_cube.coord('latitude').points), max(target_grid_cube.coord('latitude').points)),
            'longitude': (min(target_grid_cube.coord('longitude').points), max(target_grid_cube.coord('longitude').points))}
        xarray_select_uk = {k1: slice(*v1) for k1, v1 in uk_bounds.items()}

        ts_data_list = []
        # gridded_data_list = []
        for validity_time in self._target_time_range:
            self.logger.info(f'processing model data for time {validity_time}')
            single_level_ds = xarray.merge([load_ds(
                ds_path=output_dir / fname_template.format(vt=validity_time,
                                                                   lead_time=leadtime_hours,
                                                                   var_name=var1),
                selected_bounds=xarray_select_uk,
                )
                                            for var1 in sl_vars]
                                           )
            single_level_df = single_level_ds.to_dataframe().reset_index()

            height_levels_ds = xarray.merge([load_ds(
                ds_path=output_dir / fname_template.format(vt=validity_time,
                                                                   lead_time=leadtime_hours,
                                                                   var_name=var1),
                selected_bounds=xarray_select_uk,
                )
                                             for var1 in hl_vars])
            hl_df_multirow = height_levels_ds.to_dataframe().reset_index()

            var_df_merged = []
            # heights_vars_marged = height_levels_df[height_levels_df.height==heights[0]][ merge_coords]
            for var1 in height_level_var_mappings.values():
                print(var1)
                # for h1 in heights:
                #     heights_vars_marged[f'{var1}_{h1:.1f}'] = list(height_levels_df[height_levels_df.height==h1][var1])
                var_at_heights = [hl_df_multirow[hl_df_multirow.height == h1][
                                      merge_coords + [var1]].rename(
                    {var1: f'{var1}_{h1:.1f}'}, axis='columns') for h1 in heights]
                var_df_merged += [
                    functools.reduce(lambda x, y: x.merge(y, on=merge_coords),
                                     var_at_heights)]
            height_levels_df = functools.reduce(
                lambda x, y: x.merge(y, on=merge_coords), var_df_merged)

            mogreps_g_single_ts_uk_df = single_level_df.merge(height_levels_df,
                                                              on=merge_coords)
            mogreps_g_single_ts_uk_df

            mogreps_g_single_ts_uk_df = single_level_df.merge(height_levels_df,
                                                              on=merge_coords)
            ts_data_list += [mogreps_g_single_ts_uk_df]
            ts_mogg_ds1 = xarray.merge([height_levels_ds, single_level_ds])
            ts_mogg_ds1.to_netcdf(output_dir / (
                    'prd_mg_ts_' + f'{validity_time.year:04d}{validity_time.month:02d}{validity_time.day:02d}{validity_time.hour:02d}{validity_time.minute:02d}'
                    + self._fname_extension_grid)
                                  )
            # gridded_data_list += [xarray.merge([height_levels_ds, single_level_ds])]

        prd_column_dataset = pandas.concat(ts_data_list)

        fname_timestamp = self._date_fname_template.format(
            start=prd_column_dataset['time'].min(),
            end=prd_column_dataset['time'].max(), )
        model_output_fname = self._opts['model_fname_prefix'] + '_' + self._opts['leadtime_template'].format(lt=leadtime_hours) + '_' + fname_timestamp + self._fname_extension_tabular
        model_output_path = output_dir / model_output_fname
        self.logger.info(f'outputting model data to {model_output_path}')
        prd_column_dataset.to_csv(model_output_path)

        # Assign the output variable to the correct memeber variable
        # for subsequent merging
        self._merge_data = prd_column_dataset



class RadarExtractor(MassExtractor):

    def __init__(self, opts_dict):
        super().__init__(opts_dict=opts_dict)

        self.dates_to_extract = calc_dates_list(
            datetime.datetime(self._date_range[0].year,
                              self._date_range[0].month,
                              self._date_range[0].day, 0, 0),
            datetime.datetime(self._date_range[1].year,
                              self._date_range[1].month,
                              self._date_range[1].day, 23, 59),
            self._opts['archive_time_chunk'],
            )

    def extract(self):
        mass_root = self._opts['source_root']
        calc_dates_list
        fname_mass_template = self._opts['fname_mass_template']

        # radar is archived by day, we want to make sure we get the days data
        # for every day where we are looking for any part of that day




        fnames_to_extract = [fname_mass_template.format(dt=dt1)
                             for dt1 in self.dates_to_extract]

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
        inter_fname = self._opts['intermediate_fname_template']
        for pname1, p1 in self._opts['products'].items():
            for selected_day in self.dates_to_extract:
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
                        dest_root / inter_fname.format(product=pname1, selected_day=selected_day)
                        )
                iris.save(radar_day_cube, str(output_path))
                self.logger.info(f'day output to {output_path}')
                delete_file_list(radar_day_pathlist)
        self.logger.info(f'files output to {dest_root}')

    def prepare(self):
        radar_days = self.dates_to_extract
        radar_fname_template = self._opts['intermediate_fname_template']
        product1 = 'composite_rainfall'
        radar_data_dir = self._dest_path / self._opts['dataset']
        radar_cube = iris.cube.CubeList([iris.load_cube(
            str(radar_data_dir / radar_fname_template.format(selected_day=dt1,
                                                             product=product1)))
            for dt1 in radar_days]).concatenate_cube()
        validity_times = calc_dates_list(self._date_range[0],
                                         self._date_range[1],
                                         self._target_time_delta,
                                         )
        rainfall_thresholds = self._opts['rainfall_thresholds']

        # add some additional time coord info for subsequent processing
        iris.coord_categorisation.add_hour(radar_cube, coord='time')
        iris.coord_categorisation.add_day_of_year(radar_cube, coord='time')

        # load a simple cube representing the target grid
        target_grid_cube = iris.load_cube(
            str(self._target_cube_path)
        )

        coord_3hr = iris.coords.AuxCoord(radar_cube.coord('hour').points // 3,
                                        long_name='3hr',
                                         units='hour',
                                        )
        radar_cube.add_aux_coord(coord_3hr, data_dims=0)
        radar_agg_3hr = radar_cube.aggregated_by(['3hr', 'day_of_year'],iris.analysis.SUM)
        aux_coord1 = iris.coords.AuxCoord(
            [c1.bound[0] + datetime.timedelta(hours=3) for c1 in radar_agg_3hr.coord('time').cells()],
            long_name='model_accum_time',
            units='mm/h'
        )
        radar_agg_3hr.add_aux_coord(
            aux_coord1,
            data_dims=0)

        # Since we are using ionstantaneous values, which represent an hourly
        # rate, we have to divide by 12 to convert to 1 5 minute rate, which
        # we can then sum to get to our desired accumulation
        radar_agg_3hr.data = radar_agg_3hr.data * (1.0 / 12.0)

        radar_crs = radar_cube.coord_system().as_cartopy_crs()

        #TODO: bundle the creation of the auxillary lat lon coordinates into a separate function

        # Create some helper arrays for converting from our radar grid to the mogreps-g grid
        proj_y_grid = numpy.tile(radar_cube.coord('projection_y_coordinate').points.reshape(radar_cube.shape[1],1), [1, radar_cube.shape[2]])
        proj_x_grid = numpy.tile(radar_cube.coord('projection_x_coordinate').points.reshape(1,radar_cube.shape[2]), [ radar_cube.shape[1],1])

        ret_val = target_grid_cube.coord_system().as_cartopy_crs().transform_points(
            radar_crs,
            proj_y_grid,
            proj_x_grid,
            )

        lat_vals = ret_val[:,:,1]
        lon_vals = ret_val[:,:,0]

        lon_coord = iris.coords.AuxCoord(
            lon_vals,
            standard_name='longitude',
            units='degrees',
        )
        lat_coord = iris.coords.AuxCoord(
            lat_vals,
            standard_name='latitude',
            units='degrees',
        )

        radar_cube.add_aux_coord(lon_coord,[1,2])
        radar_cube.add_aux_coord(lat_coord,[1,2])
        radar_agg_3hr.add_aux_coord(lon_coord,[1,2])
        radar_agg_3hr.add_aux_coord(lat_coord,[1,2])

        # remove these coordinates as they interfere with subsequent calculations
        radar_agg_3hr.remove_coord('model_accum_time')
        radar_agg_3hr.remove_coord('forecast_reference_time')
        radar_agg_3hr.remove_coord('hour')
        radar_agg_3hr.remove_coord('day_of_year')
        radar_agg_3hr.remove_coord('3hr')

        # Calculate the latitude and longitude index in the target cube
        # coordinate system of each grid square in the radar cube.
        lat_target_index = numpy.zeros(
            (radar_cube.shape[1], radar_cube.shape[2]))
        lon_target_index = numpy.zeros(
            (radar_cube.shape[1], radar_cube.shape[2]))

        for i_lon, bnd_lon in enumerate(
                target_grid_cube.coord('longitude').bounds):

            for i_lat, bnd_lat in enumerate(
                    target_grid_cube.coord('latitude').bounds):
                arr1, arr2 = numpy.where((lat_vals >= bnd_lat[0]) &
                                         (lat_vals < bnd_lat[1]) &
                                         (lon_vals >= bnd_lon[0]) &
                                         (lon_vals < bnd_lon[1])
                                         )
                lon_target_index[arr1, arr2] = i_lon
                lat_target_index[arr1, arr2] = i_lat

        # Set up arrays to store regridded radAR precip data
        bands_agg_data = numpy.zeros(
            [len(self._target_time_range), target_grid_cube.shape[0],
             target_grid_cube.shape[1], len(rainfall_thresholds)])
        bands_instant_data = numpy.zeros(
            [len(self._target_time_range), target_grid_cube.shape[0],
             target_grid_cube.shape[1], len(rainfall_thresholds)])

        max_rain_data = numpy.zeros(
            [len(self._target_time_range), target_grid_cube.shape[0],
             target_grid_cube.shape[1]])
        mean_rain_data = numpy.zeros(
            [len(self._target_time_range), target_grid_cube.shape[0],
             target_grid_cube.shape[1]])

        # iterate through each time, rain amount band, latitude and longtitude
        for i_time, validity_time in enumerate(validity_times):
            print(validity_time)
            radar_select_time = radar_agg_3hr.extract(iris.Constraint(
                time=lambda c1: compare_time(c1.bound[0], validity_time)))
            radar_data1 = radar_select_time.data.data
            masked_radar = numpy.ma.MaskedArray(
                radar_data1,
                radar_agg_3hr[0, :, :].data.mask)

            radar_instant_select_time = radar_cube.extract(iris.Constraint(
                time=lambda c1: compare_time(c1.point, validity_time)))
            radar_instant_data1 = radar_instant_select_time.data.data
            masked_radar_instant = numpy.ma.MaskedArray(
                radar_instant_data1,
                radar_cube[0, :, :].data.mask)

            for i_lat in range(target_grid_cube.shape[0]):
                for i_lon in range(target_grid_cube.shape[1]):
                    selected_cells = (~(radar_agg_3hr[0, :, :].data.mask)) & (
                                lat_target_index == i_lat) & (
                                                 lon_target_index == i_lon)
                    masked_radar.mask = ~selected_cells
                    masked_radar_instant.mask = ~selected_cells

                    radar_cells_in_mg = numpy.count_nonzero(selected_cells)
                    # only proceed with processing for this tagret grid cell
                    # if there are some radar grid cells within this target
                    # grid cell
                    if radar_cells_in_mg > 0:
                        for imp_ix, (imp_key, imp_bounds) in enumerate(
                                rainfall_thresholds.items()):
                            # calculate fraction in band for 3 horaggregate data
                            num_in_band_agg = numpy.count_nonzero(
                                (radar_data1 >= imp_bounds[0]) & (
                                            radar_data1 <= imp_bounds[
                                        1]) & selected_cells)
                            bands_agg_data[
                                i_time, i_lat, i_lon, imp_ix] = num_in_band_agg / radar_cells_in_mg

                            # calculate raction in band for instant radar data
                            num_in_band_instant = numpy.count_nonzero(
                                (radar_instant_data1 >= imp_bounds[0]) & (
                                            radar_instant_data1 <= imp_bounds[
                                        1]) & selected_cells)
                            bands_instant_data[
                                i_time, i_lat, i_lon, imp_ix] = num_in_band_instant / radar_cells_in_mg

                        # calculate the max and average of all radar cells within each mogreps-g cell
                        max_rain_data[
                            i_time, i_lat, i_lon] = masked_radar.max()
                        mean_rain_data[i_time, i_lat, i_lon] = (
                                                                   masked_radar.sum()) / radar_cells_in_mg

                        # create instant radar rate feature data
                        max_rain_data[
                            i_time, i_lat, i_lon] = masked_radar.max()

        target_lat_coord = target_grid_cube.coord('latitude')
        target_lon_coord = target_grid_cube.coord('longitude')

        band_coord = iris.coords.DimCoord(
            [float(b1) for b1 in rainfall_thresholds.keys()],
            bounds=list(rainfall_thresholds.values()),
            var_name='band',
            units='mm',
        )
        radar_time_coord = iris.coords.DimCoord(
            [vt.timestamp() for vt in
             validity_times],
            var_name='time',
            units=radar_cube.coord('time').units,
        )
        var_name_fraction_agg = 'fraction_in_band_aggregate_3hr'
        fraction_agg_rain_band = iris.cube.Cube(
            data=bands_agg_data,
            dim_coords_and_dims=(
            (radar_time_coord, 0), (target_lat_coord, 1), (target_lon_coord, 2),
            (band_coord, 3)),
            units=None,
            var_name= var_name_fraction_agg,
            long_name='Fraction radar rainfall cells in specified rain band',
        )

        var_name_fraction_instant = 'fraction_in_band_instant'
        fraction_instant_rain_band = iris.cube.Cube(
            data=bands_instant_data,
            dim_coords_and_dims=(
            (radar_time_coord, 0), (target_lat_coord, 1), (target_lon_coord, 2),
            (band_coord, 3)),
            units=None,
            var_name=var_name_fraction_instant,
            long_name='Fraction radar rainfall cells in specified rain band',
        )

        max_rain_cube = iris.cube.Cube(
            data=max_rain_data,
            dim_coords_and_dims=(
            (radar_time_coord, 0), (target_lat_coord, 1), (target_lon_coord, 2),),
            units='mm',
            var_name='max_rain',
            long_name='maximum rain in radar cells within mogreps-g cell',
        )

        mean_rain_cube = iris.cube.Cube(
            data=mean_rain_data,
            dim_coords_and_dims=(
            (radar_time_coord, 0), (target_lat_coord, 1), (target_lon_coord, 2),),
            units='mm',
            var_name='mean_rain',
            long_name='average rain in radar cells within mogreps-g cell',
        )

        cubelist_to_save = iris.cube.CubeList(
            [fraction_agg_rain_band,
             fraction_instant_rain_band,
             max_rain_cube,
             mean_rain_cube])

        output_dir = self._dest_path
        fname_timestamp = self._date_fname_template.format(
            start=self._date_range[0],
            end=self._date_range[1],
            )
        # Save gridded radar data as a netcdf file
        grid_fname = self._opts['radar_fname_prefix'] + '_' + fname_timestamp + self._fname_extension_grid
        iris.save(cubelist_to_save, output_dir / grid_fname)

        rain_bands = list(rainfall_thresholds.keys())

        frac_agg_df = xarray.DataArray.from_iris(
            fraction_agg_rain_band).to_dataframe().reset_index()
        frac_instant_df = xarray.DataArray.from_iris(
            fraction_instant_rain_band).to_dataframe().reset_index()

        # restructure the dataframe, so that fractions in different bands are
        # separate coumns (features), rather than different data points (rows)
        radar_df = frac_agg_df[frac_agg_df['band'] == float(rain_bands[0])][
            ['time', 'latitude', 'longitude', var_name_fraction_agg]]
        radar_df = radar_df.rename(
            {var_name_fraction_agg: f'{var_name_fraction_agg}_{rain_bands[0]}'},
            axis='columns')

        for band1 in rain_bands[1:]:
            df1 = frac_agg_df[frac_agg_df['band'] == float(band1)][
                ['time', 'latitude', 'longitude', var_name_fraction_agg]]
            df1 = df1.rename({var_name_fraction_agg: f'{var_name_fraction_agg}_{band1}'},
                             axis='columns')
            radar_df = pandas.merge(radar_df, df1,
                                    on=['time', 'latitude', 'longitude'])
            df1 = frac_instant_df[frac_instant_df['band'] == float(band1)][
                ['time', 'latitude', 'longitude', var_name_fraction_instant]]
            df1 = df1.rename({var_name_fraction_instant: f'{var_name_fraction_instant}_{band1}'},
                             axis='columns')
            radar_df = pandas.merge(radar_df, df1,
                                    on=['time', 'latitude', 'longitude'])

        # merge in max and mean fields to the fraction in band fields.
        radar_df = pandas.merge(radar_df, xarray.DataArray.from_iris(
            mean_rain_cube).to_dataframe().reset_index(),
                                on=['time', 'latitude', 'longitude'])
        radar_df = pandas.merge(radar_df, xarray.DataArray.from_iris(
            max_rain_cube).to_dataframe().reset_index(),
                                on=['time', 'latitude', 'longitude'])

        radar_tab_fname = (self._opts['radar_fname_prefix'] + '_' +
                           fname_timestamp + self._fname_extension_tabular
                           )
        radar_df.to_csv(output_dir / radar_tab_fname)

        # Assign the output variable to the correct memeber variable
        # for subsequent merging
        self._merge_data = radar_df


class DummyExtractor(MassExtractor):

    def __init__(self, opts_dict, ):
        super().__init__(opts_dict=opts_dict)

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
    driver_obj = driver_class(opts_dict=init_args)
    return driver_obj

def merge_prepared_output(extractor_list, merge_vars, merge_method='inner'):
    """

    :param extractor_list:
    :param merge_vars:
    :return:
    """
    merged_df = extractor_list[0]
    if len(extractor_list) > 1:
        for extractor1 in extractor_list[1:]:
            merged_df = merged_df.merge(extractor1.data_to_merge,
                                        on=merge_vars, how=merge_method)
    return merged_df