import numpy as np
from sigmoide import *

def predMultiC(X,theta):
    # Predit la classe de chaque élement de X
    mat = np.zeros((theta.shape[0],X.shape[0]))
    for i in range(theta.shape[0]):
        mat[i] = sigmoide(np.dot(X, theta[i].T))
    
    p= np.zeros((X.shape[0],1))
    for i in range(X.shape[0]):
        p[i] = np.argmax(mat[:,i])
    
    return p