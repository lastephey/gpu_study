#!/usr/bin/env python

import os
import argparse
import datetime
import time

import numpy as np

#TODO: make broad OOP, get rid of weird module hack- make each framework, benchmark a subclass
#TODO: other benchmarks
#TODO: add ability to skip benchmarks that are not implemented
#TODO: fix docstrings
#TODO: add tmove tracking to all benchmarks

class BenchTask:

    def __init__(self, args):

        #assign values from argparse
        self.framework = args.framework
        self.benchmark = args.benchmark
        self.arraysize = args.arraysize
        self.blocksize = args.blocksize
        self.precision = args.precision
        self.ntrials = args.ntrials

        #set random seed for numpy
        np.random.seed(42)

        #create uniform input data for all frameworks to share
        #continue adding as we incorporate additional benchmarks
        if self.benchmark == 'legval':
            #legval input is 1D
            self.input_data = np.random.rand(self.arraysize).astype(self.precision)

        elif self.benchmark == 'eigh':
            #eigh input is 2D
            self.input_data = np.random.rand(self.arraysize, self.arraysize).astype(self.precision)

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
        #better way to capture date/time?
        now = datetime.datetime.now()
        self.time_filename = (location + 'timeit_{}_{}_{}_{}_{}_{}'.format(self.framework, 
                               self.benchmark, self.arraysize, self.blocksize, self.precision, now))
        return                       


    def run_kernel(self):
    
        twhole = dict()
        tmove = dict()

        #live import the module and function
        #TODO: make this nicer if possible
        module = __import__('{}_framework'.format(self.framework))
        submodule = str(self.framework) + '_' + str(self.benchmark)

        #mostly do what timeit does ourslves, but with the ability to keep results
        #and track data movement times
        for i in range(self.ntrials):
            tstart = time.time()
            #benchtask will have to record and track its own data movement time
            #for numpy (cpu) tmove will always be 0
            #TODO: find a better way to do this
            tm, results = getattr(module, submodule)(self.input_data, self.blocksize, self.precision)
            tend = time.time()
            deltat = tend-tstart
            #record trial
            twhole[i] = deltat
            tmove[i] = tm

        #save our data
        if self.framework is not 'numpy':
            np.save(self.data_filename, results)
        else:
            np.save(self.ref_filename, results)

        return tmove, twhole                            

    def correctness_check(self):
        """
    
        This function uses NumPy as ground truth data and compares other frameworks
        to the NumPy results. It uses np.allclose() to decide if the test passes 
        or fails.
    
        Input: framework, benchmark, arraysize, blocksize, precision
    
        Output: True or False, result of np.allclose
        """
    
        ref_data = np.load(self.ref_filename)
        results = np.load(self.data_filename)
    
        if self.framework is not 'numpy':
            correct = np.allclose(ref_data, results)
        else:
            correct = True

        return correct

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
    parser.add_argument('--ntrials', '-t', type=int, default=10,
                        help='how many times to run each benchmark')
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

    #print("benchtask.framework", benchtask.framework)
    #print("benchtask.benchmark", benchtask.benchmark)
    #print("benchtask.input_data.shape", benchtask.input_data.shape)

    #check to see if requested precision is possible in our framework
    #TODO: find better way to skip unimplemented benchmarks
    no64 = ['jax', 'pycuda', 'pyopencl']
    noeigh = ['numba', 'pycuda', 'pyopencl']
    if benchtask.framework in no64 and benchtask.precision == 'float64':
        print("float64 not availble in {}, skipping".format(benchtask.framework))
        return
    elif benchtask.framework in noeigh and benchtask.benchmark == 'eigh':
        print("eigh not implemented in {}, skipping".format(benchtask.framework))
        return
    else:
        tmove, twhole = benchtask.run_kernel()

        #display our min values via the min key
        tmove_key = min(tmove.keys(), key=(lambda k: tmove[k]))
        twhole_key = min(twhole.keys(), key=(lambda k: twhole[k]))
    
        print("min(tmove):", tmove[tmove_key])
        print("min(twhole):", twhole[twhole_key])
    
        correct = benchtask.correctness_check()
    
        print("correct:", correct)
    
        return

if __name__ == "__main__":
    main()


