import numpy as np

def numpy_eigh(input_data,blocksize,precision):
    """
    Here we compute eigh using numpy

    Input: input_data (shared between benchmarks), blocksize
    which in numpy is ignored
    
    Output: w, the eigenvalue matrix
    """
    #for numpy tmove=0
    tmove=0
    
    w,v = np.linalg.eigh(input_data) #w are eigenvalues, v are eigenvectors

    return tmove, w

