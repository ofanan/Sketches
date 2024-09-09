import numpy as np
import matplotlib.pyplot as plt
from settings import * 
np.set_printoptions(precision=2)

# X = [np.arange (2), np.arange (2), np.arange (2)]
X = np.array (
    [[0, 1],
     [2, 3],
     [4, 5]
    ])
epsilon = 0.5
p_val = 0.1 * np.arange (10)
y_val = np.array([1,1,1,1,1, 0,0,0,0,0])

y           = y_val
yhat        = np.where (p_val<epsilon, 1, 0)
y_bool       = np.where (y==1,    True, False)
yhat_bool    = np.where (yhat==1, True, False)
yhat_bool_not = np.where (yhat_bool==True, False, True)
y_bool_not = np.where (y_bool==True, False, True)

TP_set = y_bool * yhat_bool
TP = np.sum(np.where(TP_set==True, 1, 0))

FN_set = y_bool_not * yhat_bool
FN = np.sum(np.where(FN_set==True, 1, 0))

FP_set = y_bool * yhat_bool_not
FP = np.sum(np.where(FP_set==True, 1, 0))

prec = TP/(TP+FP)
rec  = TP/(TP+FN)

F1 = 2*prec*rec / (prec+rec)

# print (f'y={y}\nyhat={yhat}\nTP={TP}')
