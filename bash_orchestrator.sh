#!/usr/bin/env bash

#needs to be bash so we can hop in and out of conda envs

#source your favorite python
module load python

#numpy
source activate numpy
srun -n 1 python run_benchmark.py -f numpy -a 1000
source deactivate

#numba - don't think this one is getting the right answer yet
source activate numba
srun -n 1 python run_benchmark.py -f numba -a 1000
source deactivate

#cupy
source activate cupy
srun -n 1 python run_benchmark.py -f cupy -a 1000
source deactivate

#pycuda
source activate pycuda
srun -n 1 python run_benchmark.py -f pycuda -a 1000
source deactivate

#pyopencl
source activate pyopencl
srun -n 1 python run_benchmark.py -f pyopencl -a 1000
source deactivate

####jax
###source activate jax
####run stuff
###source deactivate
###
####legate
###source activate legate
####run stuff
###source deactivate

#save the data
#do the fancy analysis later in python

