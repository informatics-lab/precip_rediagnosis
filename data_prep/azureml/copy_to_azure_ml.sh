#!/usr/bin/env bash

# This script is intended to copy data from Met Office storage to Azure Blob Storage

# This should be the same directory you used as the argument --output-path when running extract_data.py
PRD_ROOT_DATASET_DIR=$1
echo Copying data from directory ${PRD_ROOT_DATASET_DIR}

# this includes the shared access token (SAS), and can be found by going to a
# container and select Shared Access Token o the left navigation bar.
PRD_AZURE_BLOB_URL=$2

# First extract files to a temprorary location
PRD_TEMP_EXTRACT_DIR=$TMPDIR/prd_transfer_$(date +%Y%m%d%H%M%S)
PRD_DATA_TO_COPY=${PRD_TEMP_EXTRACT_DIR}/prd

echo Creating temp directory to stage for transfer -  ${PRD_DATA_TO_COPY}
mkdir -p ${PRD_DATA_TO_COPY}
PRD_EXPORT_DATA_STR=prd*
echo copying data from ${PRD_DATASET_ROOT} to ${PRD_DATA_TO_COPY}

# We have to make the source directory the current directory to correctly preserve
cd ${PRD_ROOT_DATASET_DIR}
find . -iname ${PRD_EXPORT_DATA_STR}  | cpio -pdm ${PRD_DATA_TO_COPY}

# You will need to install azcopy according to the following instructions:
# https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10
cd ${PRD_TEMP_EXTRACT_DIR}
azcopy copy --recursive  prd/ ${PRD_AZURE_BLOB_URL}

#echo cleaning uup temporary directory ${PRD_TEMP_EXTRACT_DIR}
rm -rf ${PRD_TEMP_EXTRACT_DIR}


