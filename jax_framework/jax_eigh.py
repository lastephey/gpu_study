import numpy as np
import jax
from jax import jit
import jax.numpy as jnp

def jax_eigh(arraysize,blocksize):

    #create random array here for now
    x_cpu = np.random.random((arraysize,arraysize))
    x = jnp.array(x_cpu)
    
    w,v = jnp.linalg.eigh(x)
    
    #move back to cpu
    w_cpu = np.array(w)
    
    return w_cpu



