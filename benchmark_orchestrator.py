#!/usr/bin/env python

import argparse
import os
from gpu_study.pycuda import timeit_pycuda_legval
from gpu_study.numba import timeit_numba_legval
from gpu_study.cupy import timeit_cupy_legval

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
    parser.add_argument('--frameworks', '-f', type=str, default=['cupy','numba','pycuda'],
                        help='which frameworks to use')
    parser.add_argument('--benchmarks', '-b', type=str, default=['legval'],
                        help='which benchmarks to run')
    parser.add_argument('--arraysize', '-a', type=int
                        help='size of array for benchmarks')
    parser.add_argument('--blocksize', '-s', type=int, default=32,
                        help='blocksize for gpu kernels')
    parser.add_argument('--ntests', '-n', type=int, default=100,
                        help='how many times timeit will run each test')
    args = parser.parse_args()
    return args

def write_output():

    print("output data written")
    return

def main():
    args = parse_arguments()
    print("frameworks", args.frameworks)
    print("benchmarks", args.frameworks)
    print("arraysize", args.arraysize)
    print("blocksize", args.blocksize)
    print("ntests", args.ntests)
    
    #outermost loop over frameworks
    for framework in frameworks:
    
        #loop over benchmarks
        for benchmark in benchmarks:

            #generate function names
            test_name="timeit_" + framework + "_" + benchmark

            #loop over arraysizes
            for a in arraysize:

                #blocksize may or may not be a good parameter to loop over
                #some kernels really sensitive to block size, will crash with bad choice
                #loop over gpu kernel blocksize (depending on framework)
                for b in blocksize:
                    test_string="timeit_ {} _ {} .format(framework, benchmark, arraysize, )

    write_output()
 
if __name__ == "__main__":
    main()


#should save the metadata in some nice way

#should write a nice output file (or files?)





