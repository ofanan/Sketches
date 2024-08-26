import numpy as np
import matplotlib.pyplot as plt
from settings import * #error
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

x = np.array([])
print (f'x.shape={x.shape[0]}')
# m = 5
# n = 2

# X = np.array ([(i+1)*(np.arange(n)+1) for i in range(m)])
# y = np.ones (m) 
# w = np.array ([1,2])
# b = 1
# X = np.array([
#     [1,2],  # postive example
#     [3,4]])   # negative example
# W = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
# b = np.array( [-9.82, -9.28,  0.96] )
# print (f'X={X}\nw={W}\b={b}')
# print (f'X.shape={X.shape}, W.shape={W.shape}, b.shape={b.shape}')
# print (f'my_dense={my_dense(a_in=X[0], W=W, b=b)}')
# print (f'vect_dense={vectorized_dense(a_in=X[0], W=W, b=b)}')
# print (np.dot([[1,2], [3,4]], [[1,2], [2,3]]))