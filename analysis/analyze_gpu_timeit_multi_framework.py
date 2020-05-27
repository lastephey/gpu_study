#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:57:16 2020
@author: stephey
generates some plots for our gpu analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import dateutil.parser

#TODO: maybe add argparse

#for now user just enters what they need here:

directory = '/Users/stephey/Dropbox/NERSC/Work/Dates/20200521'
request_date = '2020-05-21'
#search folder for data with requested data
#by default, use most recent timestamp
benchmark = 'eigh'
frameworks = ['cupy','jax']
ntrials = 3 #hardcode for now

#we'll need this later
#messy but yolo
def get_recent_index(size, arraysize):
    for i, entry in enumerate(arraysize):
        if size == entry:
            counter = i
    return counter   

#queries results for requested terms
#will choose most recent timepoint if there are more than one 
#that meet the search criteria
def query_results(directory, request_date, benchmark, framework):

    #firstlist all files in directory
    from os import listdir
    from os.path import isfile, join
    filelist = [f for f in listdir(directory) if isfile(join(directory, f))]
    
    #filter for timeit
    timelist = [f for f in filelist if 'timeit' in f]
    
    #filter for benchmark
    benchlist = [f for f in timelist if benchmark in f]
    
    #filter for framework
    framelist = [f for f in benchlist if framework in f]
    framearray = np.array(framelist)
    
    #ok at this point, we might have several data files with
    #1) different array sizes
    #2) different time stamps for same array size
    
    #collect all arraysizes first and then look for duplicates
    #look for whatever the number is that appears after 'benchmark_'
    #ugly but yolo
    splitpoint1 = benchmark + '_'
    splitpoint2 = '_'
    
    arraysize = np.zeros(len(framelist))
    listtime = []
    for i, f in enumerate(framelist):
        
        #take the portoin after our splitpoint
        arrayend=f.split(splitpoint1,1)
        
        #now we have the arraysize
        arraysize[i] = arrayend[1].split(splitpoint2,1)[0]
        
        #now we need to get the timestamp array
        #first create intermediate datetime list, convert to numpy array next
        timepoint = dateutil.parser.parse(arrayend[1].split(splitpoint2,1)[1].strip('.npy'))
        listtime.append(timepoint)
    
    #convert list to array
    arraytime = np.array(listtime)
    
    #get indicides required to sort by time
    isort_time = np.argsort(arraytime)
    
    #now sort arraysize, arraytime, and framearray
    arraysize_sort = arraysize[isort_time]
    arraytime_sort = arraytime[isort_time]
    framearray_sort = framearray[isort_time]
    
    #get unique arraysizes
    unique_arraysize = np.unique(arraysize_sort)
    
    #use our custom function to get what we need
    index_list = []
    for size in unique_arraysize:
        ind = get_recent_index(size, arraysize_sort)
        index_list.append(ind)
    
    #keep in mind everything has to be sorted for this to work
    index_keep = np.array(index_list)    
    frame_keep = framearray_sort[index_keep]

    return frame_keep, unique_arraysize
    
#now loop over requested frameworks
#store data in dict
timeit_data = {}
timeit_min = {}
timeit_stdev = {}

#move to our requested dir
os.chdir(directory)

for framework in frameworks:    
    
    #save to assume that framework and frame_keep are in the same order? 
    frame_keep, unique_arraysize = query_results(directory, request_date, benchmark, framework)
    timeit_array = np.zeros((len(unique_arraysize), ntrials))
        
    for i, frame in enumerate(frame_keep):
        size = unique_arraysize[i]
        #there are 3 data points per frame (1 for each of 3 trials)
        timeit_array[i] = np.load(frame)
    timeit_data[framework] = timeit_array
    #find min for each arraysize
    timeit_min[framework] = np.min(timeit_data[framework],axis=1)
    #find stdev for each arraysize
    timeit_stdev[framework] = np.std(timeit_data[framework],axis=1)
  
    
#now that we're done processing the data we can finally plot    
#here we assume the same unique_arraysize - be careful
format_title = ('Frameworks: {}, Benchmark: {}').format(frameworks, benchmark)     
plt.figure()
for framework in frameworks:
    plot_ydata = timeit_min[framework]
    plot_yerror = timeit_stdev[framework]
    plot_xdata = unique_arraysize
    #plt.plot(plot_xdata, plot_ydata)
    #add errorbars based on stedev
    plt.errorbar(plot_xdata, plot_ydata, plot_yerror, marker='o')
    plt.ylabel('Min runtime (s)')
    plt.xlabel('Arraysize 1D')
plt.legend(frameworks)    
plt.title(format_title)
plt.show()


