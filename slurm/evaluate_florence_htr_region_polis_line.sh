#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J eval__florence2_htr_region__polis_line
#SBATCH -o logs_uppmax/eval__florence2_htr_region__polis_line.out
#SBATCH -e logs_uppmax/eval__florence2_htr_region__polis_line.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/visual-language-models"
cd $PROJECT_DIR

python workpad/evaluate_florence_htr_region.py \
    --input-dir $PROJECT_DIR/data/poliskammare_line