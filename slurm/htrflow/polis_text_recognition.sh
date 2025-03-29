#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 2-00:00
#SBATCH -J htrflow/polis_text_recognition
#SBATCH -o logs_uppmax/htrflow/polis_text_recognition.out
#SBATCH -e logs_uppmax/htrflow/polis_text_recognition.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/htrflow

DATA_DIR="/proj/uppmax2024-2-24/hapham/data/polis/images"
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/vlm"

cd $PROJECT_DIR

python pipelines/predict/run_htrflow_pipeline.py \
    --config-path $PROJECT_DIR/configs/htrflow/polis_text_recognition.yaml \
    --data-dir $DATA_DIR \
    --split-info-fp $PROJECT_DIR/data/polis_line/split_info.json