#!/bin/bash -l

#SBATCH -A project-name
#SBATCH -M cluster-name
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J finetune/yolo11m/sbs__region__line_od
#SBATCH -o logs_uppmax/finetune/yolo11m/sbs__region__line_od.out
#SBATCH -e logs_uppmax/finetune/yolo11m/sbs__region__line_od.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-name@email.com


source activate /crex/proj/project-name/user-name/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/user-name/vlm

cd $PROJECT_DIR

# Should train for 20 epochs
python pipelines/train/finetune_yolo.py \
    --data-dir $PROJECT_DIR/data/yolo/sbs/region__line_od \
    --data-fraction 1 \
    --base-model-path $PROJECT_DIR/models/yolo_base/yolo11m.pt \
    --model-name yolo11m__sbs__region__line_od \
    --epochs 10 \
    --batch-size 6 \
    --img-size 1280 \
