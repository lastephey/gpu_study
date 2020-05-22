#!/usr/bin/env bash

#needs to be bash so we can hop in and out of conda envs

#write logfile
LOG_FILE=/global/cscratch1/sd/stephey/gpu_study/results/log_${SLURM_JOB_ID}.out
echo $LOG_FILE
exec 3>&1 1>${LOG_FILE} 2>&1

#source your favorite python
module load python

#settings
array_size=10000
benchmark=eigh
declare -a FRAMEWORKS=("numpy" "cupy" "jax")

#loop over requested frameworks
for f in "${FRAMEWORKS[@]}"
do
    source activate $f
    printf "$f"
    srun -n 1 python run_benchmark.py -f $f -a $array_size -b $benchmark
    source deactivate
done

echo


