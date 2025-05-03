#!/bin/bash -l

#SBATCH -A uppmax2024-2-24
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -C mem512GB
#SBATCH -J data_process/create_yolo_dataset/sbs__region__line_od
#SBATCH -o logs_uppmax/data_process/create_yolo_dataset/sbs__region__line_od.out
#SBATCH -e logs_uppmax/data_process/create_yolo_dataset/sbs__region__line_od.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/vlm"

cd $PROJECT_DIR

python pipelines/data_process/create_yolo_data/create_od_dataset.py \
    --split-type sbs \
    --task region__line_od