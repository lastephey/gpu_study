#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:57:16 2020
@author: stephey
generates some plots for our gpu analysis
based heavily on https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
"""

import numpy as np
import matplotlib.pyplot as plt

#TODO: figure out a fancy way to query the data based on the current date/time.
#or have it just display the most recent results
#TODO: add way to query benchmark
#TODO: maybe add argparse

request_date = '2020-05-15'

numpy = np.load('/Users/stephey/Dropbox/NERSC/Work/Dates/20200515/timeit_numpy_legval_1000_2020-05-15 17:11:54.654629.npy')
cupy = np.load('/Users/stephey/Dropbox/NERSC/Work/Dates/20200515/timeit_cupy_legval_1000_2020-05-15 17:12:01.567798.npy')
numba = np.load('/Users/stephey/Dropbox/NERSC/Work/Dates/20200515/timeit_numba_legval_1000_2020-05-15 17:11:58.154535.npy')
pycuda = np.load('/Users/stephey/Dropbox/NERSC/Work/Dates/20200515/timeit_pycuda_legval_1000_2020-05-15 17:12:04.242708.npy')
pyopencl = np.load('/Users/stephey/Dropbox/NERSC/Work/Dates/20200515/timeit_pyopencl_legval_1000_2020-05-15 17:12:11.979813.npy')
jax = np.load('/Users/stephey/Dropbox/NERSC/Work/Dates/20200515/timeit_jax_legval_1000_2020-05-15 17:12:14.954584.npy')

frameworks = np.vstack((numpy, cupy, numba, pycuda, pyopencl, jax))

ntrials = numpy.size

trial1 = frameworks[:,0]
trial2 = frameworks[:,1]
trial3 = frameworks[:,2]

labels = ['NumPy', 'CuPy', 'Numba', 'PyCuda', 'PyOpenCL', 'JAX']


x = np.arange(len(labels))  # the label locations
width = 0.33  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, trial1, width, label='Trial 1')
rects2 = ax.bar(x, trial2, width, label='Trial 2')
rects3 = ax.bar(x + width, trial3, width, label='Trial 3', )

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_yscale('log')
ax.set_ylabel('Min runtime (s)')
ax.set_title('Python GPU Legvander')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
  

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()

