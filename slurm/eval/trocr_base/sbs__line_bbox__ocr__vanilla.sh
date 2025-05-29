#!/bin/bash -l

#SBATCH -A project-name
#SBATCH -M cluster-name
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH -J eval/trocr_base/sbs__line_bbox__ocr__vanilla
#SBATCH -o logs_uppmax/eval/trocr_base/sbs__line_bbox__ocr__vanilla.out
#SBATCH -e logs_uppmax/eval/trocr_base/sbs__line_bbox__ocr__vanilla.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-name@email.com


source activate /crex/proj/project-name/user-name/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/user-name/vlm

cd $PROJECT_DIR

python pipelines/eval/eval_trocr_ocr.py \
    --model-name trocr_base__sbs__line_bbox__ocr__vanilla \
    --data-dir $PROJECT_DIR/data/line_bbox/sbs/test/ \
    --checkpoint vanilla \
    --batch-size 10