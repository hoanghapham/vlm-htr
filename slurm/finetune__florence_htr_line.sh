#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J finetune__florence_htr_line
#SBATCH -o logs_uppmax/finetune__florence_htr_line.out
#SBATCH -e logs_uppmax/finetune__florence_htr_line.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/visual-language-models"

cd $PROJECT_DIR

python workpad/finetune_florence_htr_line.py \
    --data-dir $PROJECT_DIR/data/polis_line \
    --model-name florence-2-base-ft-htr-line \
    --train-epochs 5 \
    --use-data-pct 0.5 \
    --batch-size 2