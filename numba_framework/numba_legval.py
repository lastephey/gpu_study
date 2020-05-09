from numba import cuda
import numpy as np
import cupy as cp

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
    """Temporary wrapper that allocates memory and defines grid before calling legvander.
    Probably won't be needed once cupy has the correpsponding legvander function.
    Input: Same as cpu version of legvander
    Output: legvander matrix, cp.ndarray
    """
    x_cpu = np.random.rand(arraysize)
    deg = 10
    ideg = deg + 1
    N = x_cpu.shape[0]

    #move to gpu
    x_gpu = cp.array(x_cpu)
    v = cp.zeros((N, ideg))
    #note that this is N,ideg UNLIKE pycuda and pyopencl

    numblocks = (N + blocksize - 1) // blocksize
    legvander[numblocks, blocksize](x_gpu, deg, v)
    v_gpu = v.get()
    return v




