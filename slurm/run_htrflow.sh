#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J htrflow
#SBATCH -o logs_uppmax/htrflow.out
#SBATCH -e logs_uppmax/htrflow.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm

INPUT_DIR="/proj/uppmax2024-2-24/hapham/visual-language-models/data/poliskammare"
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/visual-language-models"

cd $PROJECT_DIR

python workpad/run_htrflow.py --input_dir $INPUT_DIR --config_path config/htr_conf.yaml