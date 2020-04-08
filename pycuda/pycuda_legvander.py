import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import sys
import numpy as np
import time

#avoid going crazy
np.random.seed(1)

#blocksize
blocksize = 16

#here are our data
x = np.random.rand(100000).astype(np.float32)
N = x.shape[0]
deg = 50
ideg = deg + 1
v = np.zeros((ideg,N)).astype(np.float32)

#print("x.shape", x.shape)
#print("v.shape", v.shape)

#allocate the gpu memory
#N_gpu = cuda.mem_alloc(4) #4 bytes for int?
#ideg_gpu = cuda.mem_alloc(4) #4 bytes for int?
x_gpu = cuda.mem_alloc(x.size * x.dtype.itemsize)
v_gpu = cuda.mem_alloc(v.size * v.dtype.itemsize)

#move the data to the gpu memory we allocated
#cuda.memcpy_htod(N_gpu, N)
#cuda.memcpy_htod(ideg_gpu, ideg)
cuda.memcpy_htod(x_gpu, x)
cuda.memcpy_htod(v_gpu, v)

####our numba legvander
###def legvander(x, deg, output_matrix):
###    i = cuda.grid(1)
###    stride = cuda.gridsize(1)
###    for i in range(i, x.shape[0], stride):
###        output_matrix[i][0] = 1
###        output_matrix[i][1] = x[i]
###        for j in range(2, deg + 1):
###            output_matrix[i][j] = (output_matrix[i][j-1]*x[i]*(2*j - 1) - output_matrix[i][j-2]*(j - 1)) / j


####the original legvander
###x = np.array(x, copy=0, ndmin=1) + 0.0
###dims = (ideg + 1,) + x.shape
###dtyp = x.dtype
###v = np.empty(dims, dtype=dtyp)
#### Use forward recursion to generate the entries. This is not as accurate
#### as reverse recursion in this application but it is more efficient.
###v[0] = x*0 + 1
###if ideg > 0:
###    v[1] = x
###    for i in range(2, ideg + 1):
###        v[i] = (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i
###return np.moveaxis(v, 0, -1)

#range is ([start], stop[, step])

#here is our cuda kernel in raw cuda
#although we are filling a 2d array (v) it is really a 1d kernel
mod = SourceModule("""
    __global__ void legvander(float *x, float *v)
    {

    int N = 100000;
    int deg = 50;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
 
    for (int i=index; i<N; i+=stride)
    {
    //remember v is a pointer, we are indexing into the 1d pointer of the 2d array v
    v[i] = 1;
    v[i+N] = x[i];

        //now we loop over row number j
        for (int j=2; j<deg +1; j++)
        {
        v[i+j*N] = (v[i+(j-1)*N] *x[i] *(2*j-1) - v[i+(j-2)*N] *(j-1)) / j;
        }

    }

    }
    """)

#get ready
func = mod.get_function("legvander")

#run our pycuda function
time_start = time.time()
func(x_gpu, v_gpu, block=(blocksize,blocksize,1))
time_end = time.time()
pycuda_time = time_end - time_start

#prepare buffer
v_result = np.zeros_like(v)
cuda.memcpy_dtoh(v_result, v_gpu)
#print("v_result:", v_result)

#try moveaxis
v_moveaxis = np.moveaxis(v_result, 0, -1)
print("v_moveaxis:", v_moveaxis)

#why do we need moveaxis here but not in numba? maybe some kind of row/column shift there?

#compare to cpu version
time_start = time.time()
v_cpu = np.polynomial.legendre.legvander(x, deg)
time_end = time.time()
cpu_time = time_end - time_start
print("v_cpu:", v_cpu)

v_diff = v_cpu - v_moveaxis
#print("v_diff:", v_diff)
v_absdiff = np.abs(v_diff / v_cpu)
print("v_absdiff:",v_absdiff)
v_diffmax = np.max(v_absdiff)
print("v_diffmax:", v_diffmax)
#different by single precision e-08 -- maybe expected since cpu is double and gpu is single?

#compare gpu and cpu time
print("pycuda time:", pycuda_time)
print("cpu time:", cpu_time)


