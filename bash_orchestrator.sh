#!/usr/bin/env bash

#needs to be bash so we can hop in and out of conda envs

#write logfile
LOG_FILE=/global/cscratch1/sd/stephey/gpu_study/results/log_${SLURM_JOB_ID}.out
echo $LOG_FILE
exec 3>&1 1>>${LOG_FILE} 2>&1

#source your favorite python
module load python

#settings
array_size=100

#numpy
source activate numpy
printf "loading numpy conda environnment\n"
srun -n 1 python run_benchmark.py -f numpy -a $array_size
source deactivate

#numba - don't think this one is getting the right answer yet
source activate numba
printf "loading numba conda environment\n"
srun -n 1 python run_benchmark.py -f numba -a $array_size
source deactivate

#cupy
source activate cupy
printf "loading cupy conda environnment\n"
srun -n 1 python run_benchmark.py -f cupy -a $array_size
source deactivate

#pycuda
source activate pycuda
printf "loading pycuda conda environment\n"
srun -n 1 python run_benchmark.py -f pycuda -a $array_size
source deactivate

#pyopencl
source activate pyopencl
printf "loading pyopencl conda environnment\n"
srun -n 1 python run_benchmark.py -f pyopencl -a $array_size
source deactivate

#jax
source activate jax
printf "loading jax conda environnment\n"
srun -n 1 python run_benchmark.py -f jax -a $array_size
source deactivate

####legate
###source activate legate
####run stuff
###source deactivate

#save the data
#do the fancy analysis later in python

