#!/bin/bash

FOLDER_PATH=$1

if [ -z "$FOLDER_PATH" ]; then
  echo "Error: Please provide the folder path as an argument"
  exit 1
fi

for file in "$FOLDER_PATH"/*; do
  sbatch $file
done