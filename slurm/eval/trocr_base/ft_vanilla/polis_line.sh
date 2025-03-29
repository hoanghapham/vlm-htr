#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J eval/trocr_base/ft_vanilla/polis_line
#SBATCH -o logs_uppmax/eval/trocr_base/ft_vanilla/polis_line.out
#SBATCH -e logs_uppmax/eval/trocr_base/ft_vanilla/polis_line.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/vlm"
cd $PROJECT_DIR

python pipelines/evaluate/evaluate_trocr_htr.py \
    --model-name trocr_base__ft_vanilla \
    --data-dir $PROJECT_DIR/data/polis_line \
    --use-split-info true \
    --batch-size 15 \
    --load-checkpoint vanilla