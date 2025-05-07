#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J finetune/florence_base/sbs__region__line_od__debug
#SBATCH -o logs_uppmax/finetune/florence_base/sbs__region__line_od__debug.out
#SBATCH -e logs_uppmax/finetune/florence_base/sbs__region__line_od__debug.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/hapham/vlm

cd $PROJECT_DIR

python pipelines/train/finetune_florence__region__line_od.py \
    --data-dir $PROJECT_DIR/data/page/sbs \
    --model-name florence_base__sbs__region__line_od__debug \
    --num-train-epochs 1 \
    --max-train-steps 14000 \
    --batch-size 2 \
    --logging-interval 7000 \
    --debug true
