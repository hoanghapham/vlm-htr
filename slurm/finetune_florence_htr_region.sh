#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 2-00:00
#SBATCH -J finetune_florence_htr_region
#SBATCH -o logs_uppmax/finetune_florence_htr_region.out
#SBATCH -e logs_uppmax/finetune_florence_htr_region.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm

PROJECT_DIR="/proj/uppmax2024-2-24/hapham/visual-language-models"

cd $PROJECT_DIR

python workpad/finetune_florence_htr_region.py --epochs 10 --use-batch-pct 1 --batch-size 2