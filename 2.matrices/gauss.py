# implementation of gauss method with max elem selection

import numpy as np

enable_logs=False

def _swap_mrx_rows(A, i1, i2):
	if i1 != i2:
		A[[i1, i2]] = A[[i2 , i1]]

def _swap_mrx_columns(A, j1, j2):
	if j1 != j2:
		A[:,[j1, j2]] = A[:,[j2, j1]]

def _swap_vec_elems(v, j1, j2):
	if j1 != j2:
		v[[j1, j2]] = v[[j2, j1]]

def _dbgs(*args, **kwargs):
	if enable_logs:
		print(*args, **kwargs)

def _dbgs_dump_Ab(A, b):
	_dbgs("A =")
	_dbgs(A)
	_dbgs("b =", b)
	_dbgs()

# dim(A) >= 2, A - square matrix
# permutes rows and columns in submatrix of A, so that elem
# with max abs value will be in position (jfrom, jfrom) and returns that value (with sign)
# Submatrix is formed with rows and columns which indices >= jfrom
# (i.e. bottom right corner of A)
# The same permutation of rows is applied to vector row_permutation
# and the same permutation of columns is applied to vector column_permutation
#
# Note. row_permutation is used to do the same permutation on the right
# side of equation Ax = b, column_permutation is used to maintain
# permutation of x[i], when columns are swaped
def permute_submatrix(A, row_permutation, column_permutation, jfrom): # -> max
	nrows, ncolumns =  A.shape
	assert nrows == ncolumns

	# seek max
	maxval = A[jfrom][jfrom]
	max_i = max_j = jfrom
	for i in range(jfrom, nrows):
		for j in range(jfrom, ncolumns):
			if abs(A[i][j]) > abs(maxval):
				maxval = A[i][j]
				max_i = i
				max_j = j
	_dbgs("selected max {} at idx {}, {}".format(maxval, max_i, max_j))
	_swap_mrx_rows(A, jfrom, max_i)
	_swap_vec_elems(row_permutation, jfrom, max_i)
	_swap_mrx_columns(A, jfrom, max_j)
	_swap_vec_elems(column_permutation, jfrom, max_j)
	return maxval

# dim(A) >= 2, A - square matrix
def solve_linear(A, b): # -> np.array x
	A = A.copy()
	b = b.copy()

	_dbgs("------------------------------")
	_dbgs("----- gauss forward pass -----")
	_dbgs("------------------------------")

	nrows, ncolumns =  A.shape
	assert nrows == ncolumns

	permutation = np.arange(ncolumns) # holds permutation of columns in matrix
	x = np.zeros(ncolumns)

	# forward pass
	for j in range(ncolumns):
		_dbgs("-- step {}:".format(j))
		maxval = permute_submatrix(A, b, permutation, j)
		_dbgs("after permutation of rows and columns:")
		_dbgs_dump_Ab(A, b)
		normalized_base_row = A[j] / maxval
		normalized_base_b = b[j] / maxval
		for i in range(j + 1, nrows):
			b[i] -= normalized_base_b * A[i][j] # doing this before A is changed in the next line
			A[i] -= normalized_base_row * A[i][j]
		_dbgs("after normalization with {}:".format(maxval))
		_dbgs_dump_Ab(A, b)

	_dbgs("------------------------------")
	_dbgs("----- gauss reverse pass -----")
	_dbgs("------------------------------")

	for i in reversed(range(nrows)):
		summ = 0
		for j in range(i + 1, ncolumns):
			assert x[j] != 0
			summ += A[i][j] * x[j]
		x[i] = (b[i] - summ) / A[i][i]
		_dbgs("found x[{}] = {}".format(i, x[i]))

	_dbgs("permuted roots:", x)
	_dbgs("permutation:", permutation)

	res = np.zeros(ncolumns)
	for i in range(ncolumns):
		res[permutation[i]] = x[i]

	_dbgs("------------------------------")
	_dbgs("result:", res)
	_dbgs("------------------------------")
	return res