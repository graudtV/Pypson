# some useful utilities

import numpy as np
from functools import reduce
from operator import add

def vec_norm_1(vec):
	return max(np.abs(elem) for elem in vec)

def vec_norm_2(vec):
	return sum(np.abs(elem) for elem in vec)

def vec_norm_3(vec):
	return np.sqrt(sum(elem ** 2 for elem in vec))

def mrx_norm_1(A):
	return max([vec_norm_2(row) for row in A])

def condition_number(A, mrx_norm=mrx_norm_1):
	return mrx_norm(A) * mrx_norm(np.linalg.inv(A))

# returns (signed) eigen value, which has the maximum abs value
def maxabs_eigen_value(A, niterations = 100):
	nrows, ncolumns =  A.shape
	assert nrows == ncolumns

	def next_iteration(A, u):
		Au = np.matmul(A, u)
		return Au / vec_norm_1(Au)

	u = np.ones(nrows)
	for i in range(niterations):
		u = next_iteration(A, u)
	return np.dot(np.matmul(A, u), u) / np.dot(u, u)

def minabs_eigen_value(A, niterations = 100):
	return 1 / maxabs_eigen_value(np.linalg.inv(A))