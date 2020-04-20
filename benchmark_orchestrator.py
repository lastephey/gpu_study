#!/usr/bin/env python

import argparse
import os

#this script should orchestrate all the benchmarks as a part of our python on gpus study

#should have nice argparse
#which benchmarks/functions
#how many timeit iterations
#arraysize
#kernelsize
#where to write what to name output? or have that happen automatically?


#reads argparse

#outer loop: which benchmark

#next loop: which framework

#next loop: arraysize

#innermost loop: blocksize

def parse_arguments():
    print("parsing data")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tests', '-t', type=str, default='all',
                        help='which kernels to run')
    parser.add_argument('--frameworks', '-f', type=str, default='all',
                        help='sum the integers (default: find the max)')
    parser.add_argument('--arraysize', '-a', type=int
                        help='size of array for benchmarks')
    parser.add_argument('--blocksize', '-b', type=int,
                        help='blocksize for gpu kernels')
    parser.add_argument('--ntests', '-n', type=int, default=10,
                        help='how many times timeit will run each test')
    args = parser.parse_args()
    return args

def launch_kernel():

    return


def write_output():

    print("output data written")
    return

def main():
    args = parse_arguments()
    print("tests", args.tests)
    print("frameworks", args.frameworks)
    print("arraysize", args.arraysize) #all unless user specifies otherwise
    print("blocksize", args.blocksize) #all unless user specifies otherwise
    print("ntests", args.ntests)
    
    #outermost loop over tests
    for test in tests:
    
        #loop over frameworks
        for framework in frameworks:

            #loop over arraysizes
            for a in arraysize:

                #blocksize may or may not be a good parameter to loop over
                #some kernels really sensitive to block size, will crash with bad choice
                #loop over gpu kernel blocksize (depending on framework)
                if framework = pycuda 
                for b in blocksize:


    write_output()
 
if __name__ == "__main__":
    main()


#should save the metadata in some nice way

#should write a nice output file (or files?)





