#!/bin/bash -l

#SBATCH -A uppmax2024-2-24
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -J eval/florence_base/sbs__line_od__ocr
#SBATCH -o logs_uppmax/eval/florence_base/sbs__line_od__ocr.out
#SBATCH -e logs_uppmax/eval/florence_base/sbs__line_od__ocr.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR="/proj/uppmax2024-2-24/hapham/vlm"

cd $PROJECT_DIR

python pipelines/eval/eval_florence_ocr.py \
    --model-name florence_base__sbs__line_od__ocr \
    --data-dir $PROJECT_DIR/data/line_bbox/sbs/test/ \
    --checkpoint best \\
    --batch-size 4