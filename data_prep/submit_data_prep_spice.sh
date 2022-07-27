#!/usr/bin/env bash

export EVENT_NAME=${1}
echo processing event ${EVENT_NAME}
export PRD_LOGS_DIR=$DATADIR/precip_rediagnosis/logs
export PRD_LOG_OUT_PATH=${PRD_LOGS_DIR}/prd_extract_data_${EVENT_NAME}.out
export PRD_LOG_ERR_PATH=${PRD_LOGS_DIR}/prd_extract_data_${EVENT_NAME}.err

echo writing spice output files to dir ${PRD_LOGS_DIR}
echo writing to log files ${PRD_LOG_OUT_PATH} and ${PRD_LOG_ERR_PATH}
sbatch -o ${PRD_LOG_OUT_PATH} -e ${PRD_LOG_ERR_PATH} --export="EVENT_NAME" data_prep/run_data_prep_spice.sh
