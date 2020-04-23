import timeit
import numpy as np

timeit_setup = 'from numba_legval import legval_kernel; arraysize=100; blocksize=32'

timeit_code = 'results = legval_kernel(arraysize, blocksize)'

repeat = 3
number = 100

times = timeit.repeat(setup=timeit_setup, stmt=timeit_code, repeat=repeat, number=number)

print('Min numba legval time of {} trials, {} runs each: {}'.format(repeat, number, min(times)))
