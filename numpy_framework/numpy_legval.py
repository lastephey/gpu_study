import numpy as np

#set numpy random seed
np.random.seed(42)

def numpy_legval(arraysize, blocksize):
    #here are our data
    x = np.random.rand(arraysize).astype(np.float32)
    N = x.shape[0]
    deg = 10
    ideg = deg + 1
    v = np.zeros((ideg,N)).astype(np.float32)
   
    #no gpu, cpu only for testing 
    
    #from the numpy source code for legvander
    v[0] = x*0 + 1
    if deg > 0:
        v[1] = x
        for i in range(2, deg + 1):
            v[i] = (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i
    #return cpu values for correctness checking
    cpu_res = np.moveaxis(v, 0, -1)
    return cpu_res


#results = numpy_legval(arraysize=100,blocksize=32)
#print(results)
