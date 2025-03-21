#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J finetune/florence_base/ft_od_region
#SBATCH -o logs_uppmax/finetune/florence_base/ft_od_region.out
#SBATCH -e logs_uppmax/finetune/florence_base/ft_od_region.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/vlm"
DATA_DIR="/proj/uppmax2024-2-24/hapham/data/"
cd $PROJECT_DIR

python pipelines/train/finetune_florence_od.py \
    --data-dir $DATA_DIR/polis \
    --model-name florence_base__ft_od_region \
    --num-train-epochs 4 \
    --max-train-steps 8000 \
    --batch-size 2 \
    --logging-interval 1000
