#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J finetune/florence_base/mixed__line_od__ocr
#SBATCH -o logs_uppmax/finetune/florence_base/mixed__line_od__ocr.out
#SBATCH -e logs_uppmax/finetune/florence_base/mixed__line_od__ocr.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2020-2-2/hapham/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/hapham/vlm

cd $PROJECT_DIR

python pipelines/train/finetune_florence_ocr.py \
    --data-dir $PROJECT_DIR/data/line_bbox/mixed/ \
    --model-name florence_base__mixed__line_od__ocr \
    --num-train-epochs 2 \
    --max-train-steps 220000 \
    --batch-size 2 \
    --logging-interval 20000 \