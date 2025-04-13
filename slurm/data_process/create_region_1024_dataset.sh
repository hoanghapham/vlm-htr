#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -C mem512GB
#SBATCH -J create_region_dataset
#SBATCH -o logs_uppmax/data_process/create_region_dataset.out
#SBATCH -e logs_uppmax/data_process/create_region_dataset.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-13/hapham/vlm"
DATA_DIR="/proj/uppmax2024-2-24/hapham/data/hovratt"

cd $PROJECT_DIR

python pipelines/data_process/create_region_dataset.py 