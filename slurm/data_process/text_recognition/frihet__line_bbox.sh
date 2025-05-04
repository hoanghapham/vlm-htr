#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -C mem512GB
#SBATCH -J data_process/create_frihetstiden__line_bbox
#SBATCH -o logs_uppmax/data_process/create_frihetstiden__line_bbox.out
#SBATCH -e logs_uppmax/data_process/create_frihetstiden__line_bbox.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2020-2-2/hapham/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/hapham/vlm

cd $PROJECT_DIR

python pipelines/data_process/create_arrow_datasets.py \
    --raw-data-dir $PROJECT_DIR/data/raw/riksarkivet/frihetstidens/ \
    --dataset-type text_recognition__line_bbox \
    --output-data-dir $PROJECT_DIR/data/processed/riksarkivet/text_recognition__line_bbox