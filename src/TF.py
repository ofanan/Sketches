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
        ### START CODE HERE ### 
        Dense (120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        Dense (40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        Dense (6, activation='linear')        
        ### END CODE HERE ### 
    ], name = "my_model" 
)

model_s.compile(
    ### START CODE HERE ### 
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    ### START CODE HERE ### 
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
