import numpy as np, torch, tensorflow as tf
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

import settings
from settings import error, warning

def compute_cost_lab (X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

def predict (X, w, b):
    
    numExamples = X.shape[0]
    ww = tf.tile ([w], [numExamples, 1])
    return tf.reduce_sum(input_tensor=tf.math.multiply (X, tf.tile ([w], [numExamples, 1])), axis=1) + b

def compute_cost_my (X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    myDtype = 'float'
    X = tf.convert_to_tensor (X, dtype=myDtype)
    y = tf.convert_to_tensor (y, dtype=myDtype)
    w = tf.convert_to_tensor (w, dtype=myDtype)
    m = X.shape[0]
    prediction = predict (X, w, b)
    costVec = tf.math.square(prediction-y)
    # print (f'prediction={prediction}\ny={y}\ncostVec={costVec}')
    # exit ()
    # cost = tf.reduce_sum(input_tensor=costVec)
    # print (f'cost={cost}')
    # exit ()
    return tf.reduce_sum(input_tensor=tf.math.square(prediction-y)) / (2*m)

X = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([ 0.29133535, 18.75376741, -53.36032453, -26.42131618])

# X = tf.tile (tf.constant([[1, 2, 3]]), [2,1])
# X = tf.constant ([[1, 2, 3], [4, 5, 6]])
# w = tf.constant ([1, 1, 1])
# ww = tf.tile ([w], [2, 1])
# prediction = tf.reduce_sum(input_tensor=tf.math.multiply (X, ww), axis=1)
# print (f'X={X}\nw={w}\nww={ww}\nprediction={prediction}')
# exit ()
# numExamples = X.shape[0]
# wMat = tf.tile ([w], [numExamples, 1])
# prediction = tf.math.multiply(X, wMat)
# error (prediction)
  
costLab = compute_cost_lab(X=X, y=y, b=b_init, w=w_init)  
costMy  = compute_cost_my (X=X, y=y, b=b_init, w=w_init)  
print (f'costLab={costLab}, costMy={costMy}')
