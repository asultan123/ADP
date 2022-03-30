#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:00
#SBATCH --job-name=seeds_train
#SBATCH --mem=32GB
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_reports/train_seeds.%j.out
#SBATCH --error=./slurm_reports/train_seeds.%j.err
#SBATCH --partition=gpu

nvidia-smi
seeds=(8388 1759 5933 7916 7445); #6130 7422 4066 3098 5469 4456 2302 9062 2724 8420);
model_dir=/scratch/sultan.a/adp_models

which python

function train_seeds()
{
    for seed in ${seeds[@]}; do
        python train_cifar.py --model_dir=$model_dir --seed=$seed --lamda=2 --log_det_lamda=0.5 --num_models=2 --augmentation=True --dataset=cifar10
    done
}

train_seeds;

exit;
