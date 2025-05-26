#!/bin/bash -l

#SBATCH -A project-name
#SBATCH -M cluster-name
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J eval/pipeline/sbs__florence__single_model_100000
#SBATCH -o logs_uppmax/eval/pipeline/sbs__florence__single_model_100000.out
#SBATCH -e logs_uppmax/eval/pipeline/sbs__florence__single_model_100000.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-name@email.com


source activate /crex/proj/project-name/user-name/envs/vlm
PROJECT_DIR=/proj/uppmax2024-2-24/user-name/vlm
cd $PROJECT_DIR

python pipelines/eval/eval_pipeline_florence__single_model.py \
    --split-type sbs \
    --checkpoint checkpoint_step_0000100000 \
    --batch-size 4