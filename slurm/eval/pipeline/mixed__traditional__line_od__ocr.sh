#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J eval/pipeline/mixed__traditional__line_od__ocr
#SBATCH -o logs_uppmax/eval/pipeline/mixed__traditional__line_od__ocr.out
#SBATCH -e logs_uppmax/eval/pipeline/mixed__traditional__line_od__ocr.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2020-2-2/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2020-2-2/hapham/vlm"
cd $PROJECT_DIR

python pipelines/eval/eval_pipeline_traditional__line_od__ocr.py \
    --split-type mixed \
    --batch-size 6