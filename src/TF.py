import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *
from public_tests import *

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

model = Sequential(
    [               
        tf.keras.Input(shape=(400,)),    #specify input size
        ### START CODE HERE ### 
        Dense (25, activation='sigmoid'),
        Dense (15, activation='sigmoid'),
        Dense (1,  activation='sigmoid')
        
        
        ### END CODE HERE ### 
    ], name = "my_model" 
)                         
# UNQ_C2
# GRADED CELL: Sequential model
tf.random.set_seed(1234) # for consistent results
# model = Sequential(
#     [               
#         ### START CODE HERE ### 
#         Dense (units=25, activation='relu'),
#         Dense (units=15, activation='relu'),
#         Dense (units=10, activation='linear'),
#
#         ### END CODE HERE ### 
#     ], name = "my_model" 
# )   
