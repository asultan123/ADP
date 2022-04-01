#!/bin/bash

function train_seeds()
{
    seeds=(8388 1759 5933 7916 7445 6130 7422 4066 3098 5469 4456 2302 9062 2724 8420);
    for seed in ${seeds[@]}; do
        echo launching job with seed $seed
	sbatch train_with_seed.sh $seed
    done
}

train_seeds;

exit;
 
