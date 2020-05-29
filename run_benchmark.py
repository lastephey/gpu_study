#!/usr/bin/env python

import os
import argparse
import timeit 
import datetime

import numpy as np

#TODO: find some way to time the data movement separately (or remove it) via separate timeit calls
#TODO: generate the data once and re-use it?
#TODO: other benchmarks
#TODO: add ability to skip benchmarks that are not implemented or precision that is not possible

class BenchTask:

    def __init__(self, args):

        #assign values from argparse
        self.framework = args.framework
        self.benchmark = args.benchmark
        self.arraysize = args.arraysize
        self.blocksize = args.blocksize
        self.precision = args.precision
        self.ntrials = args.ntrials
        self.ntests = args.ntests

        #set random seed for numpy
        np.random.seed(42)

        #continue adding as we incorporate additional benchmarks
        if self.benchmark == 'legval':
            #legval input is 1D
            self.input_data = np.random.rand(self.arraysize).astype(self.precision)

        elif self.benchmark == 'eigh':
            #eigh input is 2D
            self.input_data = np.random.rand((self.arraysize, self.arraysize)).astype(self.precision)

        else:
            print("No information for requested benchamark")
            exit()

        #hardcode path for now
        location = '/global/cscratch1/sd/stephey/gpu_study/results/'

        #establish filename for output data (used in correctness checking)
        self.data_filename = (location + str(self.framework) + '_' + str(self.benchmark)
                             + '_' + str(self.arraysize) + '_' + str(self.blocksize) + '_'
                             + (self.precision) + '.npy')

        #establish filename for the reference numpy data file
        self.ref_filename = (location + 'numpy_' + str(self.benchmark) + '_' +
                            str(self.arraysize) + '_' + str(self.blocksize) + '_' + 
                            str(self.precision) + '.npy')

        #establish filename for the timeit file
        now = datetime.datetime.now()
        self.timeit_filename = (location + 'timeit_{}_{}_{}_{}_{}_{}'.format(self.framework, 
                               self.benchmark, self.arraysize, self.blocksize, self.precision, now))
        return                       


    def time_kernel(self):
        """
        This function uses the Python timeit module to time our benchmark.
        It runs ntrials(number) times and repeat times per trial. 
    
        Input: framework, benchmark, arraysize, blocksize, precision, 
        x_input (same data for all frameworks), repeat, number
    
        Output: the array timeit_data, which contains the data from ntrials
    
        """
    
        #TODO: figure out how to reuse our same data x_input
    
        timeit_setup = 'import {}_framework; arraysize={}; blocksize={}; precision={}'\
                       .format(self.framework, self.arraysize, self.blocksize, self.precision)
        #print(timeit_setup)               
        timeit_code = 'results = {}_framework.{}_{}(arraysize, blocksize, precision)'\
                      .format(self.framework, self.framework, self.benchmark)
        #print(timeit_code)              
        timeit_list = timeit.repeat(setup=timeit_setup, stmt=timeit_code, repeat=self.ntrials, number=self.ntests)
        print('Min {} {} time of {} trials, {} runs each: {}'\
              .format(self.framework, self.benchmark, self.ntrials, self.ntests, min(timeit_list)))
    
        #can we run timeit again with the same setup? yes, here is how:
        timeit_test = timeit.repeat(setup=timeit_setup, stmt="import numpy as np", repeat=self.ntrials, number=self.ntests)
        print("numpy imported as test")
    
        timeit_data = np.array(timeit_list)
    
        #need to save timeit data in addition to printing them
        np.save(self.timeit_filename, timeit_data)
    
        return timeit_data                               

    #have to do this separately bc we can't save data from a timeit run
    def get_save_results(self):
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
    
    
        np.save(self.data_filename, results)
        return

    def correctness_check(self):
        """
    
        This function uses NumPy as ground truth data and compares other frameworks
        to the NumPy results. It uses np.allclose() to decide if the test passes 
        or fails.
    
        Input: framework, benchmark, arraysize, blocksize, precision
    
        Output: True or False, result of np.allclose
        """
    
        #TODO: incorporate precision here
    
        location = '/global/cscratch1/sd/stephey/gpu_study/results/'
    
        filename = (location + str(framework) + '_' + str(benchmark) + 
                   '_' + str(arraysize) + '_' + str(blocksize) + '.npy')
    
        results = np.load(filename,allow_pickle=True)
    
        numpy_filename = (location + 'numpy_' + str(benchmark) + '_' + 
                         str(arraysize) + '_' + str(blocksize) + '.npy')
    
        #load numpy results to compare via np.allclose               
        numpy_results = np.load(numpy_filename,allow_pickle=True)
    
        return np.allclose(results, numpy_results)


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
    parser.add_argument('--arraysize', '-a', type=int,  default=100,
                        help='size of array for benchmarks')
    parser.add_argument('--blocksize', '-k', type=int, default=32,
                        help='blocksize for gpu kernels')
    parser.add_argument('--precision', '-p', type=str, default='float64',
                        help='run benchmarks using either float32 or float64')
    parser.add_argument('--ntrials', '-t', type=int, default=3,
                        help='how many times timeit will run')
    parser.add_argument('--ntests', '-n', type=int, default=10,
                        help='how many times timeit will run each test')
    args = parser.parse_args()
    return args

def main():
    """
    This is the main functin of our run_benchmark program

    Input: command line args, which are parsed

    Output: none, although several files are saved and a logfile that
    captures stdout is written to log_$SLURMJOBID

    """

    args = parse_arguments()

    #create BenchTask object 
    benchtask = BenchTask(args)

    print("benchtask.framework", benchtask.framework)
    print("benchtask.benchmark", benchtask.benchmark)
    print("benchtask.input_data.shape", benchtask.input_data.shape)

    #now run the benchtask
    timeit_data = benchtask.time_kernel()

    print("timeit_data:", timeit_data)

###
###    #if framework = numpy. need to save the files and get them ready
###    #for correctness checking
###    get_save_results(framework, benchmark, arraysize, blocksize)
###
###    #now see if those results are actually correct (agree with numpy, anyway)
###    if framework != 'numpy':
###        correct = correctness_check(framework, benchmark, arraysize, blocksize, precision)
###    else:
###        correct = True #for numpy
###    if correct == False:
###        print('{} {} {} results do not agree with numpy'.format(framework, benchmark, precision))
###    else:
###        print('results agree')
###
###    return

if __name__ == "__main__":
    main()


