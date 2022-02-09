# Two different example solutions. Note that these are not
# the only possible valid solutions!
import numpy as np

# First: using a loop
def mat_layer_sol1(A, n):
    '''
    Returns the layer n of a matrix A,
    as a NumPy vector.
    '''
    # Find the size of the matrix
    M = A.shape[0]

    # Create an empty list
    layer = []
    
    # Loop over the row elements
    for i in range(n+1):
        # Fill elements up to and including the diagonal
        layer.append(A[n, i])
    
    # Loop over the column elements above the diagonal
    for i in range(n-1, -1, -1):
        # Fill elements above the diagonal, ascending
        layer.append(A[i, n])
    
    return np.array(layer)


# Second: using NumPy functions
def mat_layer_sol2(A, n):
    '''
    Returns the layer n of a matrix A,
    as a NumPy vector.
    '''
    # Extract the half-row (including diagonal element)
    half_row = A[n, :n+1]
    
    # Extract the half-column (excluding diagonal element)
    half_col = A[:n, n]
    
    # Reverse the half-column, concatenate into single vector
    return np.concatenate([half_row, half_col[::-1]])


# ---
# Testing
# First, choose one of the model solutions
mat_layer = mat_layer_sol1

# Test cases. We can use functions from np.testing
# Test case 1
A = np.array([[6.0, 3.0, 9.0, 10.0],
              [3.0, -1.0, 6.0, 9.0],
              [-3.0, 6.0, 1.0, -1.0],
              [5.0, 2.0, 4.0, 3.0]])
expected = np.array([3.0, -1.0, 3.0])
result = mat_layer(A, 1)

# This does nothing if the result is correct (if numbers match to 6 significant digits),
# but triggers an AssertionError if the result is incorrect.
# (Check the documentation!)
np.testing.assert_allclose(expected, result, rtol=1e-6)


# Note: the test cases will be useful to test the code you have to assess,
# but you were not required to use np.testing yourself.
# You may wish to use it in further CR tasks, however!


# Test case 2
A = np.array([[1.1, 2.2],
              [3.3, 4.4]])
expected = np.array([1.1])
result = mat_layer(A, 0)
np.testing.assert_allclose(expected, result, rtol=1e-6)

# Test case 3
A = np.array([[5.1, 6.2],
              [7.3, 8.4]])
expected = np.array([7.3, 8.4, 6.2])
result = mat_layer(A, 1)
np.testing.assert_allclose(expected, result, rtol=1e-6)

# Test case 4
A = np.array([[-9.1, 10.2, 11.3],
              [-12.4, 13.5, 14.6],
              [-15.7, 16.8, 17.9]])
expected = np.array([-15.7, 16.8, 17.9, 14.6, 11.3])
result = mat_layer(A, 2)
np.testing.assert_allclose(expected, result, rtol=1e-6)

# Test case 5: matrix of random size, full of zeros
M = np.random.randint(2, 50)
n = np.random.randint(0, M)
A = np.zeros([M, M])
expected = np.zeros([2*n + 1])
result = mat_layer(A, n)
np.testing.assert_allclose(expected, result, rtol=1e-6)

# Test case 6: matrix of random size, full of ones
M = np.random.randint(7, 80)
n = np.random.randint(0, M)
A = np.ones([M, M])
expected = np.ones([2*n + 1])
result = mat_layer(A, n)
np.testing.assert_allclose(expected, result, rtol=1e-6)

# Test case 7: random matrix -- check the first and last elements
M = np.random.randint(5, 20)
n = np.random.randint(0, M)
A = np.random.randint(-5, 6, size=[M, M])
result = mat_layer(A, n)

# The "assert" statement can also be used as above: nothing happens
# if the expression is True, but an AssertionError is raised if
# the expression is False.
assert result[0] == A[n, 0]
assert result[-1] == A[0, n]

# Choose layer 0, check that it returns only the top left element
result = mat_layer(A, 0)
assert len(result) == 1
assert result[0] == A[0, 0]

# Choose layer M-1 (last row and column), check that it matches
result = mat_layer(A, M-1)
np.testing.assert_allclose(A[-1, :], result[:M], rtol=1e-6)
np.testing.assert_allclose(A[::-1, -1], result[M-1:], rtol=1e-6)
