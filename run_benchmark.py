#!/usr/bin/env python
import argparse
import os
import timeit 
import datetime
import numpy as np

#TODO: make sure we can handle single and double precision
#TODO: find some way to time the data movement separately (or remove it) via separate timeit calls
#TODO: generate the data once and re-use it?
#TODO: other benchmarks
#TODO: add ability to skip benchmarks that are not implemented or precision that is not possible
#TODO: add more layers to bash orchestrator

def parse_arguments():
    """
    This function parses command line arguments using Python's argparse

    Input: command line arguments, passed in from bash_orchesrator.sh

    Output: args object, which contains the parsed command line arguments (and defaults)
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--framework', '-f', default=['numpy'],
                        help='which framework to use')
    parser.add_argument('--benchmark', '-b', default=['legval'],
                        help='which benchmark to run')
    parser.add_argument('--array', '-a', type=int,  default=100,
                        help='size of array for benchmarks')
    parser.add_argument('--blocksize', '-s', type=int, default=32,
                        help='blocksize for gpu kernels')
    parser.add_argument('--precision', '-p', type=str, default='double',
                        help='run benchmarks in single or double precision')
    parser.add_argument('--repeat', '-r', type=int, default=3,
                        help='how many times timeit will run')
    parser.add_argument('--ntests', '-n', type=int, default=10,
                        help='how many times timeit will run each test')
    args = parser.parse_args()
    return args

def correctness_check(framework, benchmark, arraysize, blocksize, precision):
    """

    This function uses NumPy as ground truth data and compares other frameworks
    to the NumPy results. It uses np.allclose() to decide if the test passes 
    or fails.

    Input: framework, benchmark, arraysize, blocksize, precision

    Output: True or False, result of np.allclose
    """

    #TODO: incorporate precision here

    location = '/global/cscratch1/sd/stephey/gpu_study/results/'

    filename = location + str(framework) + '_' + str(benchmark) + 
               '_' + str(arraysize) + '_' + str(blocksize) + '.npy'

    results = np.load(filename,allow_pickle=True)

    numpy_filename = location + 'numpy_' + str(benchmark) + '_' + 
                     str(arraysize) + '_' + str(blocksize) + '.npy'

    #load numpy results to compare via np.allclose               
    numpy_results = np.load(numpy_filename,allow_pickle=True)

    return np.allclose(results, numpy_results)

def get_save_results(framework, benchmark, arraysize, blocksize):
    """
    This function saves the results from the timeit runs for use
    later in correctness checking

    Input: framework, benchmark, arraysize, blocksize

    Output: none (results are saved in .npy file
    """

    #live import the module and function
    module = __import__('{}_framework'.format(framework))
    submodule = str(framework) + '_' + str(benchmark)
    results = getattr(module, submodule)(arraysize, blocksize) 

    location = '/global/cscratch1/sd/stephey/gpu_study/results/'

    filename = location + str(framework) + '_' + str(benchmark) + '_' + 
               str(arraysize) + '_' + str(blocksize) + '.npy'

    np.save(filename, results)
    return

def time_kernel(framework, benchmark, arraysize, blocksize, precision, 
                x_input, repeat, number):
    """
    This function uses the Python timeit module to time our benchmark.
    It runs ntrials(number) times and repeat times per trial. 

    Input: framework, benchmark, arraysize, blocksize, precision, 
    x_input (same data for all frameworks), repeat, number

    Output: the array timeit_data, which contains the data from ntrials

    """

    #TODO: figure out how to reuse our same data x_input

    timeit_setup = 'import {}_framework; arraysize={}; blocksize={}'\
                   .format(framework, arraysize, blocksize)
    #print(timeit_setup)               
    timeit_code = 'results = {}_framework.{}_{}(arraysize, blocksize)'\
                  .format(framework, framework, benchmark)
    #print(timeit_code)              
    timeit_list = timeit.repeat(setup=timeit_setup, stmt=timeit_code, repeat=repeat, number=number)
    print('Min {} {} time of {} trials, {} runs each: {}'\
          .format(framework, benchmark, repeat, number, min(timeit_list)))
   
    #can we run timeit again with the same setup? yes, here is how:
    timeit_test = timeit.repeat(setup=timeit_setup, stmt="import numpy as np", repeat=repeat, number=number)
    print("numpy imported as test")

    timeit_data = np.array(timeit_list)

    #need to save timeit data in addition to printing them
    now = datetime.datetime.now()
    location = '/global/cscratch1/sd/stephey/gpu_study/results/'
    timeit_filename = location + 'timeit_{}_{}_{}_{}'.format(framework, benchmark, arraysize, now)
    np.save(timeit_filename, timeit_data)

    return timeit_data

def main():
    """
    This is the main functin of our run_benchmark program

    Input: command line args, which are parsed

    Output: none, although several files are saved and a logfile that
    captures stdout is written to log_$SLURMJOBID

    """

    args = parse_arguments()

    framework = args.framework
    benchmark = args.benchmark
    arraysize = args.arraysize
    precision = args.precision
    
    print("executing {} {} arraysize {} precision {}".format(framework, benchmark, arraysize, precision))

    #TODO: depending on the benchmark, we might need to generate
    #different kinds,sizes of data. Proabaly need a more sophisticated
    #function to get this right

    if args.precision == single:
        #generate single precision array
        x_input = np.random.rand(arraysize).astype('float32')
    if args.precision == double:
        #generate double precision array
        x_input = np.random.rand(arraysize).astype('float64')
    else:
        print("Error-- precision must be single or double")
        exit()

    results = time_kernel(framework, benchmark, arraysize, blocksize,
                          precision, x_input, repeat=args.repeat, 
                          number=args.ntests)

    #if framework = numpy. need to save the files and get them ready
    #for correctness checking
    get_save_results(framework, benchmark, arraysize, blocksize)

    #now see if those results are actually correct (agree with numpy, anyway)
    if framework != 'numpy':
        correct = correctness_check(framework, benchmark, arraysize, blocksize, precision)
    else:
        correct = True #for numpy
    if correct == False:
        print('{} {} {} results do not agree with numpy'.format(framework, benchmark, precision))
    else:
        print('results agree')

    return

if __name__ == "__main__":
    main()


