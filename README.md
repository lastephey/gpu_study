# README file for python on gpus exploration

In this project we'll try lots of frameworks for using 
Python on GPUs.

# Frameworks 

## NumPy (CPU baseline and correctness)

### Legval

## CuPy

### Legval

## Numba (CUDA)

### Legval

## PyCUDA

### Legval

## PyOpenCL

### Legval

## JAX

## Legate

# Overview

The `bash_orchestrator.sh` script coordinates the benchmark. This was required
in order to be able to hop in and out of conda environments for each framework.
The `run_benchmark.py` takes care of the rest inside each conda
environment/framework. 

In each framework directory the `framework_requirements.txt` file details the
contents of each conda environment.








