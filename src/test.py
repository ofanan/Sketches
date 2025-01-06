import numpy as np

def safe_max_minus_min(arr):
    """
    Subtracts the minimum value from the maximum value of a NumPy array,
    handling potential overflow exceptions.

    Args:
        arr: A NumPy array.

    Returns:
        The difference between the maximum and minimum values, or None if
        an overflow occurs or the input is invalid.
        Also returns a boolean indicating whether overflow occurred.
    """
    if not isinstance(arr, np.ndarray):
      print("Input must be a NumPy array.")
      return None, False

    if arr.size == 0:  # Handle empty arrays
        print("Input array is empty.")
        return None, False

    try:
        max_val = np.max(arr)
        min_val = np.min(arr)
        
        # Use np.subtract with casting to a larger dtype if needed
        # to prevent overflow in many cases
        if np.issubdtype(arr.dtype, np.integer) and np.dtype(arr.dtype).itemsize < 8: #check if the array is an integer array with less than 64 bits
            result = np.subtract(max_val, min_val, dtype=np.int64) #cast to int64 to avoid overflow
        else:
            result = np.subtract(max_val, min_val)


        return result, False

    except OverflowError:
        print("Overflow occurred during subtraction.")
        return None, True
    except Exception as e: # Catch other potential exceptions (e.g., if input is not numeric)
        print(f"An unexpected error occurred: {e}")
        return None, True


# Example usage:
arr1 = np.array([1, 5, 2, 8, 3], dtype=np.int8)
result1, overflow1 = safe_max_minus_min(arr1)
print(f"Array: {arr1}, Result: {result1}, Overflow: {overflow1}")

arr2 = np.array([127, -128], dtype=np.int8) #will cause overflow if not handled correctly
result2, overflow2 = safe_max_minus_min(arr2)
print(f"Array: {arr2}, Result: {result2}, Overflow: {overflow2}")


arr3 = np.array([2**63 -1, -2**63], dtype=np.int64) #won't cause overflow because it is int64
result3, overflow3 = safe_max_minus_min(arr3)
print(f"Array: {arr3}, Result: {result3}, Overflow: {overflow3}")

arr4 = np.array([]) #empty array
result4, overflow4 = safe_max_minus_min(arr4)
print(f"Array: {arr4}, Result: {result4}, Overflow: {overflow4}")

arr5 = "not an array" #not an array
result5, overflow5 = safe_max_minus_min(arr5)
print(f"Input: {arr5}, Result: {result5}, Overflow: {overflow5}")

arr6 = np.array([1.0, 5.0, 2.0, 8.0, 3.0], dtype=np.float32) #float array
result6, overflow6 = safe_max_minus_min(arr6)
print(f"Array: {arr6}, Result: {result6}, Overflow: {overflow6}")