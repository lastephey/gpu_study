import timeit

def time_kernel(arraysize, blocksize, repeat, number):

    timeit_setup = 'from pycuda_legval import legval_kernel; arraysize={}; blocksize={}'.format(arraysize,blocksize)

    timeit_code = 'results = legval_kernel(arraysize, blocksize)'

    times = timeit.repeat(setup=timeit_setup, stmt=timeit_code, repeat=repeat, number=number)

    print('Min pycuda legval time of {} trials, {} runs each: {}'.format(repeat, number, min(times)))
