import numpy as np
from numba import cuda
import cupy as cp

#set numpy random seed
np.random.seed(42)

@cuda.jit
def legvander(x, deg, v):

    #be a little more explicit here
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x;
    stride = cuda.blockDim.x * cuda.gridDim.x;
    
    for i in range(index, x.shape[0], stride):
        v[i][0] = 1
        v[i][1] = x[i]
        for j in range(2, deg + 1):
            v[i][j] = (v[i][j-1]*x[i]*(2*j - 1) - v[i][j-2]*(j - 1)) / j

def numba_legval(arraysize, blocksize):
    #here are our data
    x_cpu = np.random.rand(arraysize)
    N = x_cpu.shape[0]
    deg = 10
    ideg = deg + 1
    v_cpu = np.ndarray((ideg,N))

    #move x and v to the device
    x = cp.asarray(x_cpu)
    v = cp.asarray(v_cpu)

    numblocks = (len(x_cpu) + blocksize - 1) // blocksize

    legvander[numblocks, blocksize](x, deg, v)

    #move v back to the host
    #v_gpu = v.copy_to_host()

    #try existing array
    #ary = np.empty(shape=v.shape, dtype=v.dtype)
    #v.copy_to_host(ary)

    #use cupy here, it's the only thing that works
    v_gpu = v.get()

    #return cpu values
    #figure out moveaxis later
    return v_gpu

#results = numba_legval(1000,32)
#print(results)
#print(results.shape)
