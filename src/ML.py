import numpy as np
import matplotlib.pyplot as plt
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    # m = x.shape[0]
    # f_wb = np.zeros(m)
    # for i in range(m):
    #     f_wb[i] = w * x[i] + b
        
    return np.array([(w*item+b) for item in x])

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0]
    Y_hat = compute_model_output (x, w, b)
    return (1/(2*m)) * np.sum(np.array([((y[i]-Y_hat[i])**2) for i in range(m)]))
    
    # x is the input variable (size in 1000 square feet)
# y is the target (price in 1000s of dollars)
x = np.array([1.0, 2.0])
y = np.array([300.0, 500.0])
m = x.shape[0]

w = 100
b = 300
print(f"w: {w}")
print(f"b: {b}")

# Plot our model prediction
plt.plot(x, compute_model_output(x, w, b,), c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x, y, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
# plt.show()

compute_cost(x, y, w, b)
