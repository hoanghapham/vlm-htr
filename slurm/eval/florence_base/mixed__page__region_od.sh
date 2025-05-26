#!/bin/bash -l

#SBATCH -A project-name
#SBATCH -M cluster-name
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH -J eval/florence_base/mixed__page__region_od
#SBATCH -o logs_uppmax/eval/florence_base/mixed__page__region_od.out
#SBATCH -e logs_uppmax/eval/florence_base/mixed__page__region_od.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-name@email.com


source activate /crex/proj/project-name/user-name/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/user-name/vlm

cd $PROJECT_DIR

python pipelines/eval/eval_florence_od.py \
    --model-name florence_base__mixed__page__region_od \
    --data-dir $PROJECT_DIR/data/page/mixed/test \
    --checkpoint best \
    --task page__region_od
