import numpy as np
import jax
from jax import jit
import jax.numpy as jnp

def jax_eigh(input_data, blocksize, precision):

    x = jnp.array(input_data).astype(precision)
    
    w,v = jnp.linalg.eigh(x)
    
    #move back to cpu
    w_cpu = np.array(w)
    
    return w_cpu



