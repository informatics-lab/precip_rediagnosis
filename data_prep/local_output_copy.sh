!#/usr/bin/env bash
FILE_EXTRACT_PATTERN="prd*"
SOURCE_DIRECTORY=$SCRATCH/precip_rediagnosis//train_202209/
TARGET_DIRECTORY=/project/informatics_lab/precip_rediagnosis/train202209/

cd ${SOURCE_DIRECTORY}
find . -iname ${FILE_EXTRACT_PATTERN} | cpio -pdm --verbose  ${TARGET_DIRECTORY}
