#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 2-00:00
#SBATCH -J finetune/trocr_base/ft_htr_line__riksarkivet
#SBATCH -o logs_uppmax/finetune/trocr_base/ft_htr_line__riksarkivet.out
#SBATCH -e logs_uppmax/finetune/trocr_base/ft_htr_line__riksarkivet.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/hapham/vlm
DATA_DIR=$PROJECT_DIR/data/riksarkivet

cd $PROJECT_DIR

python pipelines/train/finetune_trocr_htr.py \
    --data-dir $DATA_DIR \
    --model-name trocr_base__ft_htr_line__riksarkivet \
    --num-train-epochs 5 \
    --max-train-steps 10000 \
    --batch-size 10 \
    --logging-interval 1000
