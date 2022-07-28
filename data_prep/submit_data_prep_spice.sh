#!/usr/bin/env bash

export PRD_TIMESTAMP=$(date +%Y%M%dT%H%m)

export EVENT_NAME=${1}
echo processing event ${EVENT_NAME}
export PRD_JOB_NAME=prd_data_extract_${EVENT_NAME}
export PRD_LOGS_DIR=$DATADIR/precip_rediagnosis/logs
export PRD_LOG_OUT_PATH=${PRD_LOGS_DIR}/prd_extract_data_${EVENT_NAME}_${PRD_TIMESTAMP}.out
export PRD_LOG_ERR_PATH=${PRD_LOGS_DIR}/prd_extract_data_${EVENT_NAME}_${PRD_TIMESTAMP}.err

echo writing spice output files to dir ${PRD_LOGS_DIR}
echo writing to log files ${PRD_LOG_OUT_PATH} and ${PRD_LOG_ERR_PATH}
sbatch --job-name ${PRD_JOB_NAME} --output ${PRD_LOG_OUT_PATH} --error ${PRD_LOG_ERR_PATH} --export="EVENT_NAME" data_prep/run_data_prep_spice.sh
