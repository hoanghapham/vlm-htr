#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J eval/pipeline/sbs__florence__single_model_100000
#SBATCH -o logs_uppmax/eval/pipeline/sbs__florence__single_model_100000.out
#SBATCH -e logs_uppmax/eval/pipeline/sbs__florence__single_model_100000.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hoang-ha.pham.1833@student.uu.se


source activate /crex/proj/uppmax2024-2-24/hapham/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/hapham/vlm
cd $PROJECT_DIR

python pipelines/eval/eval_pipeline_florence__single_model.py \
    --split-type sbs \
    --checkpoint checkpoint_step_0000100000 \
    --batch-size 4