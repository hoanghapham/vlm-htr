#!/bin/bash -l

#SBATCH -A project-name
#SBATCH -M cluster-name
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J finetune/florence_base/sbs__region__line_od__debug
#SBATCH -o logs_uppmax/finetune/florence_base/sbs__region__line_od__debug.out
#SBATCH -e logs_uppmax/finetune/florence_base/sbs__region__line_od__debug.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-name@email.com


source activate /crex/proj/project-name/user-name/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/user-name/vlm

cd $PROJECT_DIR

python pipelines/train/finetune_florence__region__line_od.py \
    --data-dir $PROJECT_DIR/data/page/sbs \
    --model-name florence_base__sbs__region__line_od__debug \
    --num-train-epochs 1 \
    --max-train-steps 14000 \
    --batch-size 2 \
    --logging-interval 7000 \
    --debug true
