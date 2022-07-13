#!/bin/bash -l
#SBATCH --mem=96GB
#SBATCH --ntasks=8
#SBATCH --output=/scratch/shaddad/prd_debug/run_spice.out
#SBATCH --error=/scratch/shaddad/prd_debug/run_spice.err
#SBATCH --time=240

conda activate prd_data_prep
cd $HOME/prog/precip_rediagnosis/
export OUTPUT_PATH=$SCRATCH/prd_debug
python data_prep/extract_data.py --output-path ${OUTPUT_PATH} --config-file data_prep/event_data_extract.json --log-dir  ${OUTPUT_PATH} --target-cube-path /project/informatics_lab/precip_rediagnosis/target_cube.nc
