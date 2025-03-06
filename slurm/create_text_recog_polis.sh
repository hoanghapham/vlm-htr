#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J polis_text_recog
#SBATCH -o logs_uppmax/polis_text_recog.out
#SBATCH -e logs_uppmax/polis_text_recog.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm

cd $PROJECT_DIR

python workpad/create_text_recog_polis.py