import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import sys
import numpy as np

#avoid going crazy
np.random.seed(1)

#here are our data
x = np.random.rand(100)
N = x.shape[0]
deg = 10
v = np.zeros((deg+1,N))

print("x.shape", x.shape)
print("v.shape", v.shape)

#allocate the gpu memory
x_gpu = cuda.mem_alloc(x.nbytes)
v_gpu = cuda.mem_alloc(v.nbytes)

#move the data to the gpu memory we allocated
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
mod = SourceModule("""
    __global__ void legvander(float *x, float *v)
    {

    int ideg = 10;
    int N = 100;

    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int stride = blockDim.x * gridDim.x;

    // for (int i; idx<=i<N; i+=stride) {

    //here v is a pointer bc we are indexing but x is the actual data?!?!
    v[0][] = *x*0 + 1;
    v[1][] = *x;

    // we guarentee deg > 0
    // for i in range(2, ideg + 1):

    //for (int i = 2; i < deg+1; i++) {
    //    v[i] = (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i;


    }
    """)

#get ready
func = mod.get_function("legvander")

#this actually runs our kernel on the gpu
func(x_gpu, v_gpu, block=(100,1,1))

#prepare buffer
v_result = np.zeros_like(v)
cuda.memcpy_dtoh(v_result, v_gpu)
print("v:", v)
print("v_result:", v_result)

