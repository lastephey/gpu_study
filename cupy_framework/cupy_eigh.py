import numpy as np
import cupy as cp
import time

def cupy_eigh(input_data,blocksize,precision):

    tstart = time.time()
    x = cp.asarray(input_data, dtype=precision)
    tend = time.time()
    tmove = tend - tstart

    w,v = cp.linalg.eigh(x)

    #move back to cpu
    w_cpu = cp.asnumpy(w)

    return tmove, w_cpu



