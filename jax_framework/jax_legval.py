import numpy as np

import jax
from jax import jit
import jax.numpy as jnp
from jax.ops import index, index_add, index_update

#set numpy random seed
#syntax is different for jax
np.random.seed(42)

#what is the default device?
print("default device:", jax.devices()[0])

def jax_legval(arraysize, blocksize):
    #here are our data
    x_cpu = np.random.rand(arraysize).astype(np.float32)
    N = x_cpu.shape[0]
    deg = 10
    ideg = deg + 1
    v_cpu = np.zeros((ideg,N)).astype(np.float32)

    #allocate the gpu memory
    #move the data to the gpu memory we allocated
    x = jnp.array(x_cpu)
    v = jnp.array(v_cpu)

    #call jax-jitted function
    res = legvander(x, v)
    res_np = np.array(res)
    #i think we need to rotate left like the other frameworks
    res_trans = res_np.transpose(1, 0)
    return res_trans

@jax.jit
def legvander(x, v):
    #from the numpy source code for legvander
    #make it easy for now
    deg = 10
    #v[0] = x*0 + 1 #needs to be an index_update
    v = index_update(v, index[0,:], 1)
    #v[1] = x #needs to be an index_update
    v = index_update(v, index[1,:], x)
    for i in range(2, deg + 1): #i think we can get away with normal foor loops?
        #v[i] = (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i #also an index_update-- can we combine that with for loops?
        v = index_update(v, index[i,:], (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i)
    return v

###arraysize = 1000
###blocksize = 32
###results = jax_legval(arraysize, blocksize)
###print(type(results))
###print(results)
    
