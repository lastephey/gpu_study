#!/usr/bin/env python
import argparse
import os
import timeit 

#our bash_orchestrator script will call this one with the apprpropriate
#arguments for each framework

#top level needs to be be bash, not python, because we'll need a different
#codna env for each framework

#should have nice argparse
#which benchmarks/functions
#how many timeit iterations
#arraysize
#kernelsize
#where to write what to name output? or have that happen automatically?

def parse_arguments():
    print("parsing data")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--frameworks', '-f', nargs='+', default=['numpy'],
                        help='which frameworks to use')
    parser.add_argument('--benchmarks', '-b', nargs='+', default=['legval'],
                        help='which benchmarks to run')
    parser.add_argument('--arrays', '-a', type=int, nargs='+', default=100,
                        help='size of array for benchmarks')
    parser.add_argument('--blocksize', '-s', type=int, nargs='+', default=32,
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
    timeit_setup = 'import {}_framework; arraysize={}; blocksize={}'\
                   .format(framework, arraysize, blocksize)
    print(timeit_setup)               
    timeit_code = 'results = {}_framework.{}_{}(arraysize, blocksize)'\
                  .format(framework, framework, benchmark)
    print(timeit_code)              
    times = timeit.repeat(setup=timeit_setup, stmt=timeit_code, repeat=repeat, number=number)
    print('Min {} {} time of {} trials, {} runs each: {}'\
          .format(framework, benchmark, repeat, number, min(times)))

def main():
    args = parse_arguments()
    print("all frameworks:", args.frameworks)
    print("all benchmarks:", args.benchmarks)
    for framework in args.frameworks:
        for benchmark in args.benchmarks:
            for arraysize in args.arrays:
                #for blocksize in args.blocks:
                print("running benchmark:")
                print("framework:", framework)
                print("benchmark:", benchmark)
                print("arraysize:", arraysize)
                print("blocksize:", args.blocksize)
                print("repeat:", args.repeat)
                print("ntests:", args.ntests)
                results = time_kernel(framework, benchmark, arraysize, 
                                          blocksize=args.blocksize,
                                          repeat=args.repeat, number=args.ntests)
    write_output()
    return

if __name__ == "__main__":
    main()

#should save the metadata in some nice way
#should write a nice output file (or files?)
