import numpy as np

def numpy_legval(input_data, blocksize, precision):
    #for numpy tmove=0
    tmove = 0
    deg = 10
    results = np.polynomial.legendre.legvander(input_data, deg)
    return tmove, results


##for testing
#results = numpy_legval(arraysize=1000,blocksize=32)
#print(results)
#print(results.shape)
