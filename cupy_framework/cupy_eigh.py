import numpy as np
import cupy as cp

def cupy_eigh(input_data,blocksize,precision):

    x = cp.asarray(input_data, dtype=precision)

    w,v = cp.linalg.eigh(x)

    #move back to cpu
    w_cpu = cp.asnumpy(w)

    return w_cpu



