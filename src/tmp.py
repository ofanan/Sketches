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
y = y_val
y_bin = np.where (y_val==1, True, False)
yhat     = np.where (p_val<epsilon, 1, 0)
yhat_bin = np.where (p_val<epsilon, True, False)
TP_set = y_bin * yhat
TP = len (TP_set)
print (f'y={y}\nyhat={yhat}\nTP_set={TP_set}')
