#!/usr/bin/env bash

#bash lets us hop in and out of conda envs

#set to false if you want stdout
#set to true if you want to reroute to logfile
log=false

if $log; then
    LOG_FILE=/global/cscratch1/sd/stephey/gpu_study/results/log_${SLURM_JOB_ID}.out
    echo $LOG_FILE
    exec 3>&1 1>${LOG_FILE} 2>&1
fi

#source your favorite python
module load python

#array settings "" to separate items
declare -a BENCHMARKS=("legval" "eigh")
declare -a FRAMEWORKS=("numpy" "cupy" "jax" "numba" "pycuda" "pyopencl")
declare -a ARRAYSIZES=("100")
declare -a BLOCKSIZE=("32")
declare -a PRECISION=("float32" "float64")

#loop over requested frameworks
for f in "${FRAMEWORKS[@]}"
    do	
    #loop over requested benchmarks
    for b in "${BENCHMARKS[@]}"
        do
        #loop over requested arraysizes   
        for s in "${ARRAYSIZES[@]}"
            do
	    #loop over requested blocksize (will only apply to some frameworks)
            for k in "${BLOCKSIZE[@]}"
                do
                #loop over requested precision
	        for p in "${PRECISION[@]}"
	            do    
                        source activate $f
                        printf "framework: $f benchmark: $b arraysize: $s blocksize: $k precision: $p \n"
                        srun -n 1 python run_benchmark.py -f $f -b $b -a $s -k $k -p $p
                        conda deactivate
	            done    
                done
            done		
        done
    done
echo


