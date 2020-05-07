import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

#set numpy random seed
np.random.seed(42)

def pycuda_legval(arraysize, blocksize):
    #here are our data
    x = np.random.rand(arraysize).astype(np.float32)
    N = x.shape[0]
    deg = 10
    ideg = deg + 1
    v = np.zeros((ideg,N)).astype(np.float32)
    
    #allocate the gpu memory
    x_gpu = cuda.mem_alloc(x.size * x.dtype.itemsize)
    v_gpu = cuda.mem_alloc(v.size * v.dtype.itemsize)
    
    #move the data to the gpu memory we allocated
    cuda.memcpy_htod(x_gpu, x)
    cuda.memcpy_htod(v_gpu, v)
    
    #here is our cuda kernel in raw cuda
    #although we are filling a 2d array (v) it is really a 1d kernel
    mod = SourceModule("""
        __global__ void legvander(int N, int deg, float *x, float *v)
        {
    
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
    
    #define our gpu function
    func = mod.get_function("legvander")
    
    #run our pycuda function
    func(np.uint32(N), np.uint32(deg), x_gpu, v_gpu, block=(blocksize,blocksize,1))
    
    #prepare buffer
    v_result = np.zeros_like(v)
    
    #move data from host to device
    cuda.memcpy_dtoh(v_result, v_gpu)
    
    #need moveaxis to get the right answer (but not with numba)
    #why do we need moveaxis here but not in numba? maybe some kind of row/column shift there?
    v_moveaxis = np.moveaxis(v_result, 0, -1)
    #return values for correctness checking
    return v_moveaxis




