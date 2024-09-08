import numpy as np
import matplotlib.pyplot as plt
from settings import * 
np.set_printoptions(precision=2)
from fractions import Fraction
# import tensorflow as tf
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# tf.autograph.set_verbosity(0)

# Practice Lab: Decision Trees
# Ex 1
def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    if np.shape(y)[0]==0:
        entropy = 0
    else:
        p_1 = np.sum(y)/np.shape(y)[0]
        if p_1==0 or p_1==1:
            entropy = 0
        else:
            entropy = -p_1*np.log2(p_1) - (1-p_1)*np.log2(1-p_1)
    
# Ex 2: 
# def split_dataset(X, node_indices, feature):
    # left_indices    = np.where(X[node_indices,feature]==1)[0]
    # right_indices   = np.where(X[node_indices,feature]==0)[0]
    # node_indices    = np.array(node_indices)
    # left_indices    = list(map(int, node_indices[left_indices]))
    # right_indices   = list(map(int, node_indices[right_indices]))

# Ex 3
# def compute_information_gain(X, y, node_indices, feature):
    # """
    # Compute the information of splitting the node on a given feature
    #
    # Args:
    #     X (ndarray):            Data matrix of shape(n_samples, n_features)
    #     y (array like):         list or ndarray with n_samples containing the target variable
    #     node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
    #     feature (int):           Index of feature to split on
    #
    # Returns:
    #     cost (float):        Cost computed
    #
    # """    
    # # Split dataset
    # left_indices, right_indices = split_dataset(X, node_indices, feature)
    #
    # # Some useful variables
    # X_node, y_node = X[node_indices], y[node_indices]
    # X_left, y_left = X[left_indices], y[left_indices]
    # X_right, y_right = X[right_indices], y[right_indices]
    #
    # # You need to return the following variables correctly
    # information_gain = 0
    #
    # ### START CODE HERE ###
    # entropy_root = compute_entropy (y_node)
    # if len(node_indices)==0:
    #     information_gain = 0
    # else: 
    #     information_gain = entropy_root - ( (len(left_indices)/len(node_indices)) * compute_entropy (y_left) + (len(right_indices)/len(node_indices))* compute_entropy (y_right))

#Ex 4
def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    
    # Some useful variables
    num_features = X.shape[1]
    
    # You need to return the following variables correctly
    best_feature = -1
    ### START CODE HERE ###
    max_info_gain = 0
    for feature in np.arange(num_features):
        info_gain = compute_information_gain (X, y, node_indices, feature)
        if info_gain > max_info_gain:
            best_feature = feature
            max_info_gain = info_gain
    ### END CODE HERE ##    


for feature in np.arange (3):
    print (feature)
exit () 


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

# UNQ_C2
# GRADED CELL: Sequential model
tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        ### START CODE HERE ### 
        Dense (units=25, activation='relu'),
        Dense (units=15, activation='relu'),
        Dense (units=10, activation='linear'),
                
        ### END CODE HERE ### 
        
    ], name = "my_model")
               
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.build([None, 400])
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


