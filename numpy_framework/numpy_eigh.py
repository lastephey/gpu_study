import numpy as np

def numpy_eigh(arraysize,blocksize):

    #create random array here for now
    x = np.random.random((arraysize,arraysize))
    
    w,v = np.linalg.eigh(x) #w are eigenvalues, v are eigenvectors
    return w

