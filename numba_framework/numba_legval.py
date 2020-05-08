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
    x = cp.asarray(np.random.rand(arraysize))
    N = x.shape[0]
    deg = 10
    ideg = deg + 1
    v = cp.zeros((ideg,N))

    #launch the kernel
    legvander[1, blocksize](x, deg, v)
    v_cpu = cp.asnumpy(v)
    #also moveaxis here?
    #v_moveaxis = np.moveaxis(v_cpu, 0, -1)
    return v_cpu


#for testing
results = numba_legval(1000,32)
print(results)
print(results.shape)
