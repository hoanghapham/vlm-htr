#!/bin/bash -l

#SBATCH -A project-name
#SBATCH -M cluster-name
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J eval/pipeline/time__mixed__florence__region_od__line_od__ocr
#SBATCH -o logs_uppmax/eval/pipeline/time__mixed__florence__region_od__line_od__ocr.out
#SBATCH -e logs_uppmax/eval/pipeline/time__mixed__florence__region_od__line_od__ocr.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-name@email.com


source activate /crex/proj/project-name/user-name/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/user-name/vlm
cd $PROJECT_DIR

python pipelines/eval/time_pipeline_florence__region_od__line_od__ocr.py \
    --split-type mixed \
    --batch-size 2