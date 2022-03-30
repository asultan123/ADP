#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:00
#SBATCH --job-name=seeds_train
#SBATCH --mem=32GB
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --output=myjob_seeds.%j.out
#SBATCH --error=myjob_seeds.%j.err
#SBATCH --partition=gpu

nvidia-smi
# conda activate tf
# cd /home/sultan.a/Adaptive-Diversity-Promoting

seeds=(8388 1759 5933 7916 7445 6130 7422 4066 3098 5469 4456 2302 9062 2724 8420);

function echo_seeds()
{
    for seed in ${seeds[@]}; do
        echo $seed; 
    done
}

echo_seeds;
exit;
