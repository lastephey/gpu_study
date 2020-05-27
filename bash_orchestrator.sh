#!/usr/bin/env bash

#bash lets us hop in and out of conda envs

#write logfile
LOG_FILE=/global/cscratch1/sd/stephey/gpu_study/results/log_${SLURM_JOB_ID}.out
echo $LOG_FILE
exec 3>&1 1>${LOG_FILE} 2>&1

#source your favorite python
module load python

#settings
benchmark=eigh
declare -a FRAMEWORKS=("numpy" "cupy" "jax")
declare -a ARRAYSIZES=("100" "200" "500" "1000" "2000" "5000")

#loop over requested frameworks
for f in "${FRAMEWORKS[@]}"
    #loop over requested arraysizes
    do    
    for s in "${ARRAYSIZES[@]}"
        do
            source activate $f
            printf "$f"
            srun -n 1 python run_benchmark.py -f $f -a $s -b $benchmark
            source deactivate
        done
    done
echo


