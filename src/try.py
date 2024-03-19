import os, scipy, matplotlib, numpy as np
# from datetime import datetime
from tictoc import tic, toc
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
line1, = ax.plot([1, 2, 3], label="test1")
line2, = ax.plot([3, 2, 1], label="test2")
legend1 = ax.legend(handles=[line1], loc='upper right')
ax.add_artist(legend1)

ax.legend(handles=[line2], loc='lower left')
plt.show()