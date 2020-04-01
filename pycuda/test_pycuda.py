# code directly copied from https://wiki.tiker.net/PyCuda/Examples/Demo

# Sample source code from the Tutorial Introduction in the documentation.

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy
a = numpy.random.randn(4,4)

a = a.astype(numpy.float32)

#have to manually allocate the memmroy
a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

#have to manually move the memory
cuda.memcpy_htod(a_gpu, a)

#here is our cuda kernel in raw cuda
mod = SourceModule("""
    __global__ void doublify(float *a)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      a[idx] *= 2;
    }
    """)

#not sure what this does
func = mod.get_function("doublify")

#this actually runs our kernel on the gpu
func(a_gpu, block=(4,4,1))

#preallocate buffer to move data back
a_doubled = numpy.empty_like(a)

#manually move the data back to the cpu
cuda.memcpy_dtoh(a_doubled, a_gpu)
print("original array:")
print(a)
print("doubled with kernel:")
print(a_doubled)

# alternate kernel invocation -------------------------------------------------

func(cuda.InOut(a), block=(4, 4, 1))
print("doubled with InOut:")
print(a)

# part 2 ----------------------------------------------------------------------

import pycuda.gpuarray as gpuarray
a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
a_doubled = (2*a_gpu).get()

print("original array:")
print(a_gpu)
print("doubled with gpuarray:")
print(a_doubled)


