#!/usr/bin/env bash
FILE_EXTRACT_PATTERN="prd*"
SOURCE_DIRECTORY=$1
TARGET_DIRECTORY=$2

cd ${SOURCE_DIRECTORY}
mkdir -p ${TARGET_DIRECTORY}
find . -iname ${FILE_EXTRACT_PATTERN} | cpio -pdm --verbose  ${TARGET_DIRECTORY}