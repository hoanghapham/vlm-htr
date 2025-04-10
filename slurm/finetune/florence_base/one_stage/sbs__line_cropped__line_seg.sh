#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J finetune/florence_base/sbs__line_cropped__line_seg
#SBATCH -o logs_uppmax/finetune/florence_base/sbs__line_cropped__line_seg.out
#SBATCH -e logs_uppmax/finetune/florence_base/sbs__line_cropped__line_seg.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/vlm"

cd $PROJECT_DIR

python pipelines/train/finetune_florence__line_cropped__line_seg.py \
    --data-dir $PROJECT_DIR/data/page/sbs/ \
    --model-name florence_base__sbs__line_cropped__line_seg \
    --model-dirs /proj/uppmax2024-2-13/hapham/vlm/training \
    --num-train-epochs 2 \
    --max-train-steps 220000 \
    --batch-size 2 \
    --logging-interval 20000 \