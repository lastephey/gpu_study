import numpy as np
import cupy as cp
import time

def cupy_legval(input_data, blocksize, precision):

    N = input_data.shape[0]
    deg = 10
    ideg = deg + 1
    v_cpu = np.zeros((ideg,N)).astype(precision)
    
    #allocate the gpu memory 
    #move the data to the gpu memory we allocated
    tstart = time.time()
    x = cp.asarray(input_data)
    v = cp.asarray(v_cpu)
    tend = time.time()
    tmove = tend - tstart

    #here is our cuda kernel in raw cuda
    #although we are filling a 2d array (v) it is really a 1d kernel
    
    #from the numpy source code for legvander
    v[0] = x*0 + 1
    if deg > 0:
        v[1] = x
        for i in range(2, deg + 1):
            v[i] = (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i
    #return cpu values
    cpu_res = v.get()
    #need to transpose to get in the same form as numpy
    cpu_trans = cpu_res.transpose(1, 0)
    return tmove, cpu_trans

##for testing
#results = cupy_legval(arraysize=1000,blocksize=32)
#print(results)
#print(results.shape)

