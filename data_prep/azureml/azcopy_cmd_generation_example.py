import os
import pathlib

srs_token = 'sp=racwdl&st=2022-03-29T10:18:49Z&se=2022-03-29T18:18:49Z&spr=https&sv=2020-08-04&sr=c&sig=iqJpmlYXG%2FHltQDumgVcn3k3xxxQ2HRAVyQ1iSAG84k%3D'
root_url = 'https://preciprediagnosisstorage.blob.core.windows.net/prd-storm-dennis'
src_dir = 'mogreps-radar'
file_list = ['composite_quality_20200214.nc',
'composite_quality_20200215.nc',
'composite_quality_20200216.nc',
'composite_quality_20200217.nc',
'composite_quality_20200218.nc',
'composite_rainfall_20200214.nc',
'composite_rainfall_20200215.nc',
'composite_rainfall_20200216.nc',
'composite_rainfall_20200217.nc',
'composite_rainfall_20200218.nc',
'prd_mogreps_g_015H_20200215T0300Z_20200217T0900Z.csv',
'prd_mogreps_g_015H_20200215T0300Z_20200217T0900Z.nc',
'prd_radar_20200214T0127Z_20200218T2227Z.nc']

cmd_template_copy = 'azcopy copy "{src}" "{dest}"'
cmd_template_delete = 'azcopy remove "{path}"'
output_path = pathlib.Path(os.environ['HOME']) / 'prog' / 'precip_rediagnosis' / 'copy_cmds.sh'

cmd_list = []
for f1 in file_list:
    src_path = root_url + '/' + src_dir + '/' + f1 + '?' + srs_token
    dest_path = root_url + '/' + f1 + '?' + srs_token
    cmd_list += [cmd_template_copy.format(src=src_path, dest=dest_path) ]
    cmd_list += [cmd_template_delete.format(path=src_path)]

with open(output_path,'w') as cmd_file:
    cmd_file.writelines((c1 + '\n' for c1 in cmd_list))