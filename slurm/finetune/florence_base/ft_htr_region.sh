#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -C mem512GB
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J finetune/florence_base/ft_htr_region
#SBATCH -o logs_uppmax/finetune/florence_base/ft_htr_region.out
#SBATCH -e logs_uppmax/finetune/florence_base/ft_htr_region.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/vlm"

cd $PROJECT_DIR

python pipelines/train/finetune_florence_htr.py \
    --data-dir $PROJECT_DIR/data/polis_region \
    --model-name florence_base__ft_htr_region \
    --train-epochs 30 \
    --use-data-pct 1 \
    --batch-size 2