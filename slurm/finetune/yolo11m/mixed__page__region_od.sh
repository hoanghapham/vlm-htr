#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J finetune/yolo11m/mixed__page__region_od
#SBATCH -o logs_uppmax/finetune/yolo11m/mixed__page__region_od.out
#SBATCH -e logs_uppmax/finetune/yolo11m/mixed__page__region_od.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/vlm"

cd $PROJECT_DIR

# Should train for 20 epochs
python pipelines/train/finetune_yolo_od.py \
    --data-dir $PROJECT_DIR/data/yolo/mixed/region_od \
    --data-fraction 1 \
    --base-model-path $PROJECT_DIR/models/yolo_base/yolo11m.pt \
    --model-name yolo11m__mixed__page__region_od \
    --epochs 20 \
    --batch-size 6 \
    --img-size 1280 \
