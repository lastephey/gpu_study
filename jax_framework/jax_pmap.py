#let's test out the jax pmap functionality

from jax import random, pmap, lax
import jax.numpy as jnp

# Create n random 5000 x 6000 matrices, one per GPU
n = 2
keys = random.split(random.PRNGKey(0), n)

print("keys", keys)

mats = pmap(lambda key: random.normal(key, (5000, 6000)))(keys)

# Run a local matmul on each device in parallel (no data transfer)
result = pmap(lambda x: jnp.dot(x, x.T))(mats)  # result.shape is (8, 5000, 5000)

# Compute the mean on each device in parallel and print the result
print(pmap(jnp.mean)(result))
# prints [1.1566595 1.1805978 ... 1.2321935 1.2015157]

#try to all_gather
y = pmap(lambda x: lax.all_gather(x, 'i'), axis_name='i')(result)

print("all_gather:")
print(y.shape)
