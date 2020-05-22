import numpy as np
import cupy as cp

def cupy_eigh(arraysize,blocksize):

    #create random array here for now
    x_cpu = np.random.random((arraysize,arraysize))
    x = cp.asarray(x_cpu)

    w,v = cp.linalg.eigh(x)

    #move back to cpu
    w_cpu = cp.asnumpy(w)

    return w_cpu



