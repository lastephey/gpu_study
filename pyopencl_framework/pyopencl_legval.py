#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

#set numpy random seed
np.random.seed(42)

#try to make sure we get a gpu
devices = cl.get_platforms()[0].get_devices(cl.device_type.GPU)
print(devices)

def pyopencl_legval(arraysize, blocksize):
    #here are our data
    x_cpu = np.random.rand(arraysize).astype(np.float32)
    N = x_cpu.shape[0]
    deg = 10
    ideg = deg + 1
    v_cpu = np.zeros((ideg,N)).astype(np.float32)

    #create context and queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    #allocate the gpu memory
    mf = cl.mem_flags
    x = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_cpu)
    v = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v_cpu)

    #need to figure out how to pass constants in here, too
    prg = cl.Program(ctx, """
    __kernel void legvander(
        ushort const N, ushort const deg,
        __global const float *x, __global float *v)
        {

        // int index = blockIdx.x * blockDim.x + threadIdx.x;
        // int stride = blockDim.x * gridDim.x;

        // blockDim.x = get_local_size(0), number of work-items per work-group or cuda: threads in block 
        // gridDim.x = get_num_groups(0), total number of work-groups or cuda: number of blocks in grid

        int index = get_global_id(0); //x-coord- (1) is y-coord
        int stride = get_local_size(0)*get_num_groups(0);

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
        """).build()
   
    #now actually run the gpu kernel
    #passing in command queue = queue
    #global size = x.shape
    #local work size = none (up to implementation)
    #special way to handle constants- np.uint16()
    #opencl buffers for kernel parameters
    prg.legvander(queue, x_cpu.shape, None, np.uint16(N), np.uint16(deg), x, v)

    #move the data back to the cpu
    v_gpu = np.empty_like(v_cpu)
    cl.enqueue_copy(queue, v_gpu, v)
   
    #need moveaxis here, same as pyopencl
    v_moveaxis = np.moveaxis(v_gpu, 0, -1)
    #return values for correctness checking
    return v_moveaxis

#results = pyopencl_legval(arraysize=100,blocksize=32)
#print(results)


