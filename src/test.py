import numpy as np

original_array = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Get the indices that would sort the array
sorted_indices = np.argsort(original_array)

# Get the sorted array using the indices
sorted_array = original_array[sorted_indices]

print("Original Array:", original_array)
print("Sorted Indices:", sorted_indices)
print("Sorted Array:", sorted_array)