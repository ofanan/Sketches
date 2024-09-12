import numpy as np
import matplotlib.pyplot as plt
from settings import * 
np.set_printoptions(precision=2)

# # Course 3 (unsupervised), week 1
# # X = [np.arange (2), np.arange (2), np.arange (2)]
# X = np.array (
#     [[0, 1],
#      [2, 3],
#      [4, 5]
#     ])
# epsilon = 0.5
# p_val = 0.1 * np.arange (10)
# y_val = np.array([1,1,1,1,1, 0,0,0,0,0])
#
# y           = y_val
# yhat        = np.where (p_val<epsilon, 1, 0)
# y_bool       = np.where (y==1,    True, False)
# yhat_bool    = np.where (yhat==1, True, False)
# yhat_bool_not = np.where (yhat_bool==True, False, True)
# y_bool_not = np.where (y_bool==True, False, True)
#
# TP_set = y_bool * yhat_bool
# TP = np.sum(np.where(TP_set==True, 1, 0))
#
# FN_set = y_bool_not * yhat_bool
# FN = np.sum(np.where(FN_set==True, 1, 0))
#
# FP_set = y_bool * yhat_bool_not
# FP = np.sum(np.where(FP_set==True, 1, 0))
#
# prec = TP/(TP+FP)
# rec  = TP/(TP+FN)
#
# F1 = 2*prec*rec / (prec+rec)

# # # Course 3 (unsupervised), week 2, lab 1: collaborative filtering
#     ### START CODE HERE ###      
#     for j in range(nu):
#         for i in range (nm):
#             if R[i][j]==0:
#                 continue          
#             J += np.square (np.dot (W[j], X[i]) + b[0][j] - Y[i][j])
#     J += (lambda_ * (np.sum (np.square(W)) + np.sum(np.square(X))))   
#     J /= 2

X = np.array (
    [[0, 1],
     [1, 1],
     [4, 5],
     [0, 0]
    ])
centroids = np.array([
    [0, 0], 
    [4, 5]]
)
K = centroids.shape[0]

idx     = np.zeros(X.shape[0], dtype=int) # idx[j] will hold the idx of the closest centroid to item i
minDist = float('inf')*np.ones (X.shape[0]) # dists[j] will hold the distance from item j to the closest centroid 
    
    ### START CODE HERE ###
for centroidNum in range(K):
    distFromThisCentroid = np.sum(np.square(X - centroids[centroidNum, :]), 1)
    idx = np.where (distFromThisCentroid<minDist, centroidNum, idx)
    minDist = np.where (minDist < distFromThisCentroid, minDist, distFromThisCentroid) 

print (f'idx={idx}')
m, n = X.shape

# You need to return the following variables correctly
centroids = np.zeros((K, n))

### START CODE HERE ###
for centroidIdx in range(K):
    pointsOfThisCentroid = np.where (idx==centroidIdx)
    points = X [pointsOfThisCentroid, :]
    centroids[centroidIdx, :] = np.mean (points, axis=1)
    # print (f'centroidIdx={centroidIdx}, pointsOfThisCentroid={pointsOfThisCentroid}\npoints={points}')
    # print (f'centroid={np.mean (points, axis=1)}')
