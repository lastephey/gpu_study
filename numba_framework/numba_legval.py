from numba import cuda
import numpy as np
import cupy as cp
import time

@cuda.jit
def legvander(x, deg, v):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(i, x.shape[0], stride):
        v[i][0] = 1
        v[i][1] = x[i]
        for j in range(2, deg + 1):
            v[i][j] = (v[i][j-1]*x[i]*(2*j - 1) - v[i][j-2]*(j - 1)) / j

def numba_legval(input_data, blocksize, precision):
    deg = 10
    ideg = deg + 1
    N = input_data.shape[0]

    #move to gpu
    tstart = time.time()
    x_gpu = cp.array(input_data)
    v = cp.zeros((N, ideg))
    tend = time.time()
    tmove = tend - tstart
    #note that this is N,ideg UNLIKE pycuda and pyopencl

    numblocks = (N + blocksize - 1) // blocksize
    legvander[numblocks, blocksize](x_gpu, deg, v)
    v_gpu = v.get()
    return tmove, v

#####for testing
####x = np.random.rand(100)
####results = numba_legval(x, 32, 'float32')
####print(results)




