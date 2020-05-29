import numpy as np

def numpy_legval(input_data, blocksize):
    deg = 10
    results = np.polynomial.legendre.legvander(input_data, deg)
    return results

##for testing
#results = numpy_legval(arraysize=1000,blocksize=32)
#print(results)
#print(results.shape)
