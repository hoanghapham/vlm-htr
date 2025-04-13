#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -C mem512GB
#SBATCH -J data_process/create_inst_seg_lines_within_regions_yolo__sbs
#SBATCH -o logs_uppmax/data_process/create_inst_seg_lines_within_regions_yolo__sbs.out
#SBATCH -e logs_uppmax/data_process/create_inst_seg_lines_within_regions_yolo__sbs.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-13/hapham/vlm"

cd $PROJECT_DIR

python pipelines/data_process/create_yolo_dataset__inst_seg_lines_within_regions.py \
    --split-type sbs