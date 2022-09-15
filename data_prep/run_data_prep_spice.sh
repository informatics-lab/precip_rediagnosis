#!/bin/bash -l
#SBATCH --mem=96GB
#SBATCH --ntasks=8
#SBATCH --time=240

#export EVENT_NAME=${1}
#export EVENT_NAME=2020_storm_dennis
echo Processing event ${EVENT_NAME}

export SRC_PATH=$HOME/prog/precip_rediagnosis/
export OUTPUT_PATH=$SCRATCH/precip_rediagnosis/train_202209
export EVENT_CONFIG=${SRC_PATH}/data_prep/event_configs/${EVENT_NAME}.json
export TARGET_CUBE_PATH=/project/informatics_lab/precip_rediagnosis/target_cube.nc

conda activate prd_data_prep
cd ${SRC_PATH}

python data_prep/extract_data.py --output-path ${OUTPUT_PATH} --config-file ${EVENT_CONFIG} --log-dir  ${OUTPUT_PATH}/logs --target-cube-path ${TARGET_CUBE_PATH}
