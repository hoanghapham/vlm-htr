#!/bin/bash -l

#SBATCH -A project-name
#SBATCH -M cluster-name
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J finetune/florence_base/sbs__page__line_od
#SBATCH -o logs_uppmax/finetune/florence_base/sbs__page__line_od.out
#SBATCH -e logs_uppmax/finetune/florence_base/sbs__page__line_od.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-name@email.com


source activate /crex/proj/project-name/user-name/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/user-name/vlm

cd $PROJECT_DIR

python pipelines/train/finetune_florence_od.py \
    --data-dir $PROJECT_DIR/data/page/sbs/ \
    --model-name florence_base__sbs__page__line_od \
    --num-train-epochs 10 \
    --max-train-steps 40000 \
    --batch-size 2 \
    --logging-interval 4000 \
    --detect-class line
