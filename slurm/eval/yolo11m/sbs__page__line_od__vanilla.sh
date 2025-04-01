#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J eval/yolo11m/sbs__page__line_od__vanilla
#SBATCH -o logs_uppmax/eval/yolo11m/sbs__page__line_od__vanilla.out
#SBATCH -e logs_uppmax/eval/yolo11m/sbs__page__line_od__vanilla.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/vlm"
cd $PROJECT_DIR

python pipelines/eval/eval_yolo_od.py \
    --data-dir $PROJECT_DIR/data/pages/sbs/test \
    --model-name yolo11m__sbs__page__line_od__vanilla \
    --checkpoint vanilla \
    --batch-size 10 \
    --object-class line \