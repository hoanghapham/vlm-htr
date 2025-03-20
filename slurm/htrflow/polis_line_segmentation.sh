#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 2-00:00
#SBATCH -J htrflow/polis_line_segmentation
#SBATCH -o logs_uppmax/htrflow/polis_line_segmentation.out
#SBATCH -e logs_uppmax/htrflow/polis_line_segmentation.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm

INPUT_DIR="/proj/uppmax2024-2-24/hapham/vlm/data/polis"
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/vlm"

cd $PROJECT_DIR

python pipelines/predict/run_htrflow_pipeline.py \
    --input_dir $INPUT_DIR \
    --split-info-fp $PROJECT_DIR/data/polis_line/split_info.json \
    --config_path $PROJECT_DIR/configs/htrflow/polis_line_segmentation.yaml