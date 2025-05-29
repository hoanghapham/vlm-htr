#!/bin/bash -l

#SBATCH -A project-name
#SBATCH -M cluster-name
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J eval/florence_base/sbs__line_od__ocr__line_od__140000
#SBATCH -o logs_uppmax/eval/florence_base/sbs__line_od__ocr__line_od__140000.out
#SBATCH -e logs_uppmax/eval/florence_base/sbs__line_od__ocr__line_od__140000.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-name@email.com


source activate /crex/proj/project-name/user-name/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/user-name/vlm

cd $PROJECT_DIR

python pipelines/eval/eval_florence_od.py \
    --model-name florence_base__sbs__line_od__ocr \
    --data-dir $PROJECT_DIR/data/page/sbs/test/ \
    --output-dir $PROJECT_DIR/evaluations/florence_base__sbs__line_od__ocr__line_od/checkpoint_step_0000140000/ \
    --task page__line_od \
    --checkpoint checkpoint_step_0000140000 \
    --batch-size 4