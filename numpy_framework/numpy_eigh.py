import numpy as np

def numpy_eigh(input_data,blocksize):
    """
    Here we compute eigh using numpy

    Input: input_data (shared between benchmarks), blocksize
    which in numpy is ignored
    
    Output: w, the eigenvalue matrix
    """
    
    w,v = np.linalg.eigh(input_data) #w are eigenvalues, v are eigenvectors

    return w

