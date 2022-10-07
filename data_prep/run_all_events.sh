#!/usr/bin/env bash
export PRD_OUTPUT_DIR=$1
echo Writing data to ${PRD_OUTPUT_DIR}
# This script should be run from the Data Prep Environment
for f1 in $(ls -1 data_prep/event_configs/*.json); do
  EVENT_NAME=$(python -c "import pathlib;print(pathlib.Path('${f1}').stem)")
  echo ${EVENT_NAME}
  ./data_prep/submit_data_prep_spice.sh ${EVENT_NAME} ${PRD_OUTPUT_DIR}
done;
