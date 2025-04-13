#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J finetune/yolo11m/mixed__line_cropped__line_seg
#SBATCH -o logs_uppmax/finetune/yolo11m/mixed__line_cropped__line_seg.out
#SBATCH -e logs_uppmax/finetune/yolo11m/mixed__line_cropped__line_seg.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-13/hapham/vlm"

cd $PROJECT_DIR

# Should train for 20 epochs
python pipelines/train/finetune_yolo.py \
    --data-dir /proj/uppmax2024-2-13/hapham/vlm/data/yolo/mixed/line_cropped__line_seg \
    --data-fraction 1 \
    --base-model-path $PROJECT_DIR/models/yolo_base/yolo11m-seg.pt \
    --model-name yolo11m_seg__mixed__line_cropped__line_seg \
    --epochs 1 \
    --batch-size 2 \
    --img-size 1280 \
    --resume false