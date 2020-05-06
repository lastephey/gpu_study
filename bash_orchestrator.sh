#!/usr/bin/env bash

#needs to be bash so we can hop in and out of conda envs

#source your favorite python
module load python

#numpy
source activate benchmark_numpy
#run stuff
source deativate

#cupy
source activate benchmark_cupy
#run stuff
source deactivate

#pycuda
source activate benchmark_pycuda
#run stuff
source deactivate

#pyopencl
source activate benchmark_pyopencl
#run stuff
source deactivate

#jax
source activate benchmark_jax
#run stuff
source deactivate

#legate
source activate benchmark_legate
#run stuff
source deactivate

#save the data
#do the fancy analysis later in python

