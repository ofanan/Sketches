import copy, math, numpy as np
import matplotlib.pyplot as plt
from utils import *
import settings
from settings import error, warning
np.set_printoptions(precision=4)

def sigmoid(z):
    return np.power(1 + np.exp (-z), -1)

def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost 
    """

    m, n = X.shape
    
    ### START CODE HERE ### 
    # z=np.squeeze(np.dot(X,w[:,np.newaxis]))
    z = np.dot(X,w) + b
    f = sigmoid (z)
    # f = np.round(sigmoid (z=np.squeeze(np.dot(X,w[:,np.newaxis]))), 3)
    # EPSILON = 10**(-18)
    # oneMinusF = np.maximum (1-f, EPSILON*np.ones(f.shape[0]))
    # print (f'oneMinusF={oneMinusF}')
    # loss = -y*np.log(f) - (1-y) * np.log (oneMinusF)
    oneMinusF = 1 - f
    print (f'X.shape={X.shape}, y.shape={y.shape}, w.shape={w.shape}, z.shape={z.shape}, f.shape={f.shape}')
    # print (f'X={X}\nw={w}\nz={z}\ny={y}') 
    loss = - y * np.log(f) - (1-y) * np.log (oneMinusF)
    total_cost = np.sum(loss)/m 
    total_cost = 7
    return total_cost

def compute_gradient(X, y, w, b, *argv): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    z = np.dot(X,w) + b
    f = sigmoid (z)
    fMinusY = f - y
    dj_db = np.sum(fMinusY)/m
    dj_dw = np.dot (fMinusY, X) / m 

def predict(X, w, b): 
    """
    """
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)    
    
    z = np.dot(X,w) + b
    f = sigmoid (z)
    p = np.where (f>0.5, 1, 0)
    print (f'f.shape={f.shape}\nf={f}\np={p}')
    
# UNQ_C1
# GRADED CELL: my_softmax

def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    ### START CODE HERE ###
    expVec = np.exp(z) 
    a = expVec / np.sum(expVec) 
    # denominator = np.sum (expVec)
    # print (denominator)
    
    
        
    
        
    
    
    ### END CODE HERE ### 
    return a
# m = 5
# n = 2
#
# rng = np.random.default_rng()
# # X = rng.random ([m, n])
# X = np.array ([(i+1)*(np.arange(n)+1) for i in range(m)])
# y = np.ones (m) #np.ones (m)
# w = np.array ([1,2])
# b = 1
#
# predict (X, w, b)


m = 5
n = 1
z = np.array ([(i+1)*(np.arange(n)+1) for i in range(m)])
print (z)
print (my_softmax(z))