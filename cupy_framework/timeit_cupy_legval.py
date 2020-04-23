import timeit

timeit_setup = 'from cupy_legval import legval_kernel; arraysize=1000; blocksize=16'

timeit_code = 'results = legval_kernel(arraysize, blocksize)'

repeat = 3
number = 100

times = timeit.repeat(setup=timeit_setup, stmt=timeit_code, repeat=repeat, number=number)

print('Min cupy legval time of {} trials, {} runs each: {}'.format(repeat, number, min(times)))
