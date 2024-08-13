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
    return tf.reduce_sum(input_tensor=X*w+b, axis=1)

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
    cost = 0.0
    prediction = predict (X, w, b)
    error (prediction)
    print (f'costs shape={costs.shape}')
    yy = tf.tile ([y], [1, m])
    print (f'y={y}, yy={yy}')
    # print (f'yy shape={yy.shape}')
    # print (f'costsb after y={costs}')
    exit ()
    for i in range(m):                                
        cost = cost + (np.dot(X[i], w) + b - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

# gamad = tf.constant ([1, 2, 3])
# nanas = tf.tile ([gamad], [2, 1])
# nanasSum = tf.reduce_sum(input_tensor=nanas, axis=1)
# print(f'nanas={nanas}\nsum={nanasSum}')
X = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
  
costLab = compute_cost_lab(X=X, y=y, b=b_init, w=w_init)  
costMy  = compute_cost_my (X=X, y=y, b=b_init, w=w_init)  
print (f'costLab={costLab}, costMy={costMy}')
