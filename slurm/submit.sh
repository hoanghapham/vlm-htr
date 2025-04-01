#!/bin/bash

# Read file paths from standard input (pipe)
while read -r file_path; do
    # Check if file exists before submitting
    if [[ -f "$file_path" ]]; then
        echo "Submitting: $file_path"
        sbatch "$file_path"
    else
        echo "Warning: File not found - $file_path"
    fi
done