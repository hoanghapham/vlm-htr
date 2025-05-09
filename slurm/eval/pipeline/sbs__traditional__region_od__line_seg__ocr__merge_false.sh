#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J eval/pipeline/sbs__traditional__region_od__line_seg__ocr__merge_false
#SBATCH -o logs_uppmax/eval/pipeline/sbs__traditional__region_od__line_seg__ocr__merge_false.out
#SBATCH -e logs_uppmax/eval/pipeline/sbs__traditional__region_od__line_seg__ocr__merge_false.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/hapham/vlm
cd $PROJECT_DIR

python pipelines/eval/eval_pipeline_traditional__region_od__line_seg__ocr.py \
    --split-type sbs \
    --sort-mode consider_margins \
    --batch-size 6 \
    --merge false