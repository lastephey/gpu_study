import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

#avoid going crazy
np.random.seed(1)

#here are our data
x = np.random.rand(100)
N = x.shape[0]
deg = 10
v = np.zeros_like(x)

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
    __global__ void legvander(float *x, int *N, int *deg, float *v)
    {

    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int stride = blockDim.x * gridDim.x;

    // for (int i; idx<=i<N; i+=stride) {

    v[0] = 1;
    // v[1] = x;
    // we guarentee deg > 0
    // for i in range(2, ideg + 1):

    //for (int i = 2; i < deg+1; i++) {
    //    v[i] = (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i;


    }
    """)

#get ready
func = mod.get_function("legvander")

#this actually runs our kernel on the gpu
func(cuda.InOut(x), cuda.InOut(N), cuda.InOut(deg), cuda.InOut(v), block=(100,100,1))

print("v:")
print(v)

#### alternate kernel invocation -------------------------------------------------
###
####this option will do the data transfer for you, maybe that's better
###
###func(cuda.InOut(a), block=(4, 4, 1))
###print("doubled with InOut:")
###print(a)
###
#### part 2 ----------------------------------------------------------------------
###
###import pycuda.gpuarray as gpuarray
###a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
###a_doubled = (2*a_gpu).get()
###
###print("original array:")
###print(a_gpu)
###print("doubled with gpuarray:")
###print(a_doubled)


