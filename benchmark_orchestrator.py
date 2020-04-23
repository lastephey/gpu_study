#!/usr/bin/env python
import argparse
import os
import timeit 

#Project/
#  Animals/
#    __init__.py
#    Mammals.py
#    Birds.py\

#in __init__.py

#from .Mammals import Mammals
#from .Birds import Birds

import numpy_framework
#from pycuda_framework.pycuda_legval import legval_kernel
#from numba_framework.numba_legval import legval_kernel
#from cupy_framework.cupy_legval import legval_kernel

#this script should orchestrate all the benchmarks as a part of our python on gpus study

#should have nice argparse
#which benchmarks/functions
#how many timeit iterations
#arraysize
#kernelsize
#where to write what to name output? or have that happen automatically?

def parse_arguments():
    print("parsing data")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--frameworks', '-f', type=str, default=['numpy'],
                        help='which frameworks to use')
    parser.add_argument('--benchmarks', '-b', type=str, default=['legval'],
                        help='which benchmarks to run')
    parser.add_argument('--arrays', '-a', type=int, default=[100,1000],
                        help='size of array for benchmarks')
    parser.add_argument('--blocks', '-s', type=int, default=[32,64],
                        help='blocksize for gpu kernels')
    parser.add_argument('--repeat', '-r', type=int, default=3,
                        help='how many times timeit will run')
    parser.add_argument('--ntests', '-n', type=int, default=100,
                        help='how many times timeit will run each test')
    args = parser.parse_args()
    return args

def write_output():
    print("output data written")
    return

def time_kernel(framework, benchmark, arraysize, blocksize, repeat, number):
    timeit_setup = 'from {}_{} import {}_kernel; arraysize={}; blocksize={}'\
                   .format(framework,benchmark,benchmark,arraysize,blocksize)
    print(timeit_setup)               
    timeit_code = 'results = {}_kernel(arraysize, blocksize)'\
                  .format(benchmark)
    print(timeit_code)              
    times = timeit.repeat(setup=timeit_setup, stmt=timeit_code, repeat=repeat, number=number)
    print('Min {} {} time of {} trials, {} runs each: {}'\
          .format(framework, benchmark, repeat, number, min(times)))

def main():
    args = parse_arguments()
    print("frameworks", args.frameworks)
    print("benchmarks", args.frameworks)
    print("arraysize", args.arrays)
    print("blocksize", args.blocks)
    print("repeat", args.repeat)
    print("ntests", args.ntests)
    for framework in args.frameworks:
        for benchmark in args.benchmarks:
            for arraysize in args.arrays:
                for blocksize in args.blocks:
                    #blocksize may or may not be a good parameter to loop over

                    results = time_kernel(framework, benchmark, arraysize, 
                                          blocksize, repeat=args.repeat, 
                                          number=args.ntests)
    write_output()
    return

if __name__ == "__main__":
    main()

#should save the metadata in some nice way
#should write a nice output file (or files?)
