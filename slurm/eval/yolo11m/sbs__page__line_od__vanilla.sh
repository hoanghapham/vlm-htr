#!/bin/bash -l

#SBATCH -A project-name
#SBATCH -M cluster-name
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J eval/yolo11m/sbs__page__line_od__vanilla
#SBATCH -o logs_uppmax/eval/yolo11m/sbs__page__line_od__vanilla.out
#SBATCH -e logs_uppmax/eval/yolo11m/sbs__page__line_od__vanilla.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-name@email.com


source activate /crex/proj/project-name/user-name/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/user-name/vlm
cd $PROJECT_DIR

python pipelines/eval/eval_yolo_od.py \
    --data-dir $PROJECT_DIR/data/page/sbs/test \
    --model-name yolo11m__sbs__page__line_od__vanilla \
    --checkpoint vanilla \
    --batch-size 10 \
    --task page__line_od \