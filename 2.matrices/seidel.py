# implementation of seidel method

import numpy as np
import math_utils

enable_logs = False

def solve_linear(A, b, precision):
	nrows, ncolumns =  A.shape
	assert nrows == ncolumns

	x = np.zeros(ncolumns)
	while math_utils.vec_norm_1(np.matmul(A, x) - b) > precision:
		for i in range(ncolumns):
			sum1 = sum(A[i][j] * x[j] for j in range(i))
			sum2 = sum(A[i][j] * x[j] for j in range(i + 1, ncolumns))
			x[i] = (b[i] - sum1 - sum2) / A[i][i]
	return x