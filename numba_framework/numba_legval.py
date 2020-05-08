import numpy as np
import cupy as cp
from numba import cuda

#set numpy random seed
np.random.seed(42)

@cuda.jit
def legvander(x, deg, v):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(i, x.shape[0], stride):
        v[i][0] = 1
        v[i][1] = x[i]
        for j in range(2, deg + 1):
            v[i][j] = (v[i][j-1]*x[i]*(2*j - 1) - v[i][j-2]*(j - 1)) / j

def numba_legval(arraysize, blocksize):
    #here are our data
    x = np.asarray(np.random.rand(arraysize)).astype(np.float32)
    N = x.shape[0]
    deg = 10
    ideg = deg + 1
    v = cp.ndarray((ideg,N)).astype(cp.float32)

    numblocks = (len(x) + blocksize - 1) // blocksize
    legvander[numblocks, blocksize](x, deg, v)
    return v


#for testing
results = numba_legval(1000,32)
print(results)
print(results.shape)
