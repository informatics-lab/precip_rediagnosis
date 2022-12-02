# source this file to run interacely.
# You should do this from in an interactive spice session created using salloc

# set this to the filename of the event of interest
#export EVENT_NAME=2020_storm_ciara.json
export EVENT_NAME=${1}
echo Processing event ${EVENT_NAME}

export SRC_PATH=$HOME/prog/precip_rediagnosis/
export OUTPUT_PATH=/scratch/shaddad/precip_rediagnosis/train_202212
export EVENT_CONFIG=${SRC_PATH}/data_prep/event_configs/${EVENT_NAME}.json
export TARGET_CUBE_PATH=/project/informatics_lab/precip_rediagnosis/target_cube.nc

conda activate prd_data_prep
cd ${SRC_PATH}

# I recommend sourcing this file with the line below commented out, then calling it manually outside the file, as you'll likely want to call it multiple times while debugging
# python data_prep/extract_data.py --output-path ${OUTPUT_PATH} --config-file ${EVENT_CONFIG} --log-dir  ${OUTPUT_PATH}/logs --target-cube-path ${TARGET_CUBE_PATH}
