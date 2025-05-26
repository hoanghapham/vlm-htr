#!/bin/bash -l

#SBATCH -A project-name
#SBATCH -M cluster-name
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J finetune/florence_base/mixed__line_bbox__ocr
#SBATCH -o logs_uppmax/finetune/florence_base/mixed__line_bbox__ocr.out
#SBATCH -e logs_uppmax/finetune/florence_base/mixed__line_bbox__ocr.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-name@email.com


source activate /crex/proj/project-name/user-name/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/user-name/vlm

cd $PROJECT_DIR

python pipelines/train/finetune_florence_ocr.py \
    --data-dir $PROJECT_DIR/data/line_bbox/mixed/ \
    --model-name florence_base__mixed__line_bbox__ocr \
    --num-train-epochs 2 \
    --max-train-steps 220000 \
    --batch-size 2 \
    --logging-interval 20000 \