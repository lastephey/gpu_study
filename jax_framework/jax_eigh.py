import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
import time

def jax_eigh(input_data, blocksize, precision):

    tstart = time.time()
    x = jnp.array(input_data).astype(precision)
    tend = time.time()
    tmove = tend - tstart

    w,v = jnp.linalg.eigh(x)
    
    #move back to cpu
    w_cpu = np.array(w)
    
    return tmove, w_cpu



