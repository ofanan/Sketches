import numpy as np
import matplotlib.pyplot as plt
from settings import * 
np.set_printoptions(precision=2)
from fractions import Fraction
# import tensorflow as tf
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# tf.autograph.set_verbosity(0)

def sigmoid(z):
    return np.power(1 + np.exp (-z), -1)

def vectorized_dense(a_in, W, b):
    print (f'a_in.shape={a_in.shape}, W.shape={W.shape}')
    print (f'dot={np.dot (a_in, W)}\nmatMul={np.matmul(a_in, W)}')
    return sigmoid(np.matmul (a_in, W) + b)

def my_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):               
        w = W[:,j]                                    
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)               
    return(a_out)

m           = 5 # of examples
n           = 3 # of dimensions
K           = 2 # of centroids
X           = np.ones  ([m, n])
centroids   = 0.9*np.ones ([K, n]) 

# print (f'X={X}\ncentroids={centroids}')

# idx     = np.zeros(X.shape[0], dtype=int)
# minDist = float('inf') * np.ones (X.shape[0])
# for centroidNum in range(K):
#     dist = np.linalg.norm(X - centroids [centroidNum, :], axis=1)
#     idx = np.where (dist<minDist, centroidNum, idx)
#     minDist = np.minimum (dist, minDist)
#
# for centroidNum in range(K):
#     pointsOfThisCentroid = np.where (idx==centroidNum)[0]
#     centroids[centroidNum] = np.mean(X[pointsOfThisCentroid], axis=0)
# print (centroids)
# a = np.array ([1,1,1])
# b = np.array ([1,2,3])
# print (np.divide (a, b))
    # print (pointsOfThisCentroid)
    # centroids[centroidNum] = np.mean (X[np.where (idx==centroidNum)[0]])
    # print (np.mean(X[pointsOfThisCentroid], axis=0))
# minDist = np.array([2, 2, 2])
# dist    = np.array([1, 2, 3])
# print (np.where (dist>1))
# print (np.minimum (dist, minDist))
# print (np.where (dist<minDist, dist, minDist))


