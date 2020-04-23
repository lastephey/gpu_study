import numpy as np
import cupy as cp

def legval_kernel(arraysize, blocksize):
    #here are our data
    x_cpu = np.random.rand(arraysize).astype(np.float32)
    N = x_cpu.shape[0]
    deg = polydeg
    ideg = deg + 1
    v_cpu = np.zeros((ideg,N)).astype(np.float32)
    
    #allocate the gpu memory 
    #move the data to the gpu memory we allocated
    x = cp.asarray(x_cpu)
    v = cp.asarray(v_cpu)
    
    #here is our cuda kernel in raw cuda
    #although we are filling a 2d array (v) it is really a 1d kernel
    
    #from the numpy source code for legvander
    v[0] = x*0 + 1
    if deg > 0:
        v[1] = x
        for i in range(2, deg + 1):
            v[i] = (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i
    #return cpu values for correctness checking
    cpu_res = cp.moveaxis(v, 0, -1).get()
    return cpu_res

