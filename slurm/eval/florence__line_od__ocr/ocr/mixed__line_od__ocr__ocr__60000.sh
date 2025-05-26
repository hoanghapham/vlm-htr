#!/bin/bash -l

#SBATCH -A project-name
#SBATCH -M cluster-name
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J eval/florence_base/mixed__line_od__ocr__ocr_60000
#SBATCH -o logs_uppmax/eval/florence_base/mixed__line_od__ocr__ocr_60000.out
#SBATCH -e logs_uppmax/eval/florence_base/mixed__line_od__ocr__ocr_60000.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-name@email.com


source activate /crex/proj/project-name/user-name/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/user-name/vlm

cd $PROJECT_DIR

python pipelines/eval/eval_florence_ocr.py \
    --model-name florence_base__mixed__line_od__ocr \
    --data-dir $PROJECT_DIR/data/line_bbox/mixed/test/ \
    --output-dir $PROJECT_DIR/evaluations/florence_base__mixed__line_od__ocr__ocr/checkpoint_step_0000060000/ \
    --checkpoint checkpoint_step_0000060000 \
    --batch-size 4