#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J data_process/create_bergskollegium_rel_line_dataset
#SBATCH -o logs_uppmax/data_process/create_bergskollegium_rel_line_dataset.out
#SBATCH -e logs_uppmax/data_process/create_bergskollegium_rel_line_dataset.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
INPUT_DIR="/proj/uppmax2024-2-24/hapham/data/riksarkivet/bergskollegium_rel"
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/vlm"

cd $PROJECT_DIR

python pipelines/data_process/create_line_dataset.py \
    --input-dir $INPUT_DIR \
    --output-dir $PROJECT_DIR/data/riksarkivet/bergskollegium_rel_line