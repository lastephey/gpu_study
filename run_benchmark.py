#!/usr/bin/env python
import argparse
import os
import timeit 
import datetime
import numpy as np

#TODO: look at precision between pycuda, pyopencl, and numpy
#TODO: find some way to time the data movement separately (or remove it) via separate timeit calls
#TODO: can we generate the data once and re-use it?
#TODO: other benchmarks
#TODO: make bash orchestrator more clever

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
    parser.add_argument('--ntests', '-n', type=int, default=10,
                        help='how many times timeit will run each test')
    args = parser.parse_args()
    return args

def correctness_check(framework, benchmark, arraysize, blocksize):
    print("checking for correctness")
    location = '/global/cscratch1/sd/stephey/gpu_study/results/'
    filename = location + str(framework) + '_' + str(benchmark) + '_' + str(arraysize) + '_' + str(blocksize) + '.npy'
    results = np.load(filename)
    numpy_filename = location + 'numpy_' + str(benchmark) + '_' + str(arraysize) + '_' + str(blocksize) + '.npy'
    numpy_results = np.load(numpy_filename)
    return np.allclose(results, numpy_results) #will have to adjust the default tolerance here

def get_save_results(framework, benchmark, arraysize, blocksize):
    #use this function to compare our framework to numpy cpu baseline (which we'll take as ground truth)
    #need some good naming conventions, good way to save files so we can compare the right thing
    print("computing results")

    #some hacks to live import the module and function
    module = __import__('{}_framework'.format(framework))
    submodule = str(framework) + '_' + str(benchmark)
    results = getattr(module, submodule)(arraysize, blocksize) 
    print("{} results: {}".format(framework, results))
    print(results.shape)

    #now save the data (eventually add timestamp, jobid, something better...)
    location = '/global/cscratch1/sd/stephey/gpu_study/results/'
    filename = location + str(framework) + '_' + str(benchmark) + '_' + str(arraysize) + '_' + str(blocksize) + '.npy'
    np.save(filename, results)
    print("results data saved")
    return

def time_kernel(framework, benchmark, arraysize, blocksize, repeat, number):
    #add array creation and data movement to setup?

    timeit_setup = 'import {}_framework; arraysize={}; blocksize={}'\
                   .format(framework, arraysize, blocksize)
    #print(timeit_setup)               
    timeit_code = 'results = {}_framework.{}_{}(arraysize, blocksize)'\
                  .format(framework, framework, benchmark)
    #print(timeit_code)              
    timeit_list = timeit.repeat(setup=timeit_setup, stmt=timeit_code, repeat=repeat, number=number)
    print('Min {} {} time of {} trials, {} runs each: {}'\
          .format(framework, benchmark, repeat, number, min(timeit_list)))
   
    #can we run timeit again with the same setup?
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
    args = parse_arguments()
    print("all frameworks:", args.frameworks)
    print("all benchmarks:", args.benchmarks)
    for framework in args.frameworks:
        for benchmark in args.benchmarks:
            for arraysize in args.arrays:
                #for blocksize in args.blocks:
                blocksize = args.blocksize
                print("running benchmark:")
                print("framework:", framework)
                print("benchmark:", benchmark)
                print("arraysize:", arraysize)
                print("blocksize:", blocksize)
                print("repeat:", args.repeat)
                print("ntests:", args.ntests)
                results = time_kernel(framework, benchmark, arraysize, blocksize,
                                          repeat=args.repeat, number=args.ntests)

                #if framework = numpy. need to save the files and get them ready
                #for correctness checking via allclose or something (although most likely
                #we'll only agree to single precision in some frameworks)
                get_save_results(framework, benchmark, arraysize, blocksize)
                #now see if those results are actually correct (agree with numpy, anyway)
                if framework != 'numpy':
                    correct = correctness_check(framework, benchmark, arraysize, blocksize)
                else:
                    correct = True #for numpy
                if correct == False:
                    print('{} {} results do not agree with numpy'.format(framework, benchmark))
                else:
                    print('results agree')

    return

if __name__ == "__main__":
    main()


