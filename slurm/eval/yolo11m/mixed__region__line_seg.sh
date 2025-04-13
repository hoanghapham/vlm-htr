#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J eval/yolo11m_seg/mixed__region__line_seg
#SBATCH -o logs_uppmax/eval/yolo11m_seg/mixed__region__line_seg.out
#SBATCH -e logs_uppmax/eval/yolo11m_seg/mixed__region__line_seg.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/vlm"
cd $PROJECT_DIR

python pipelines/eval/eval_yolo_seg.py \
    --data-dir $PROJECT_DIR/data/yolo/mixed/inst_seg_lines_within_regions/test \
    --model-name yolo11m_seg__mixed__region__line_seg \
    --checkpoint best \
    --batch-size 6 \
    --device cpu