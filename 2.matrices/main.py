# task II.10.6.п

import gauss
import seidel
import math_utils
import numpy as np

# set this to True if you want to see A, b, x
dump_matrices = False

def create_matrices(): # -> A, b
	n = 99
	A = np.zeros((n + 1, n + 1))

	for i in range(n):
		A[i][i] = 10 + (i + 1)
		A[i][i + 1] = 1
		A[i + 1][i] = 1
	for j in range(1, n + 1): # 1, 2, ..., n
		A[n][j] = 2
	A[n][0] = A[n][n] = 1

	b = np.fromfunction(lambda i: (i + 1) / n, (n + 1,))
	return A, b

A, b = create_matrices()
if dump_matrices:
	print("A =")
	print(A)
	print("B =", b)

print("condition number of A: ", math_utils.condition_number(A))
print("max eigen value of A: ", math_utils.maxabs_eigen_value(A))
print("min eigen value of A: ", math_utils.minabs_eigen_value(A))

x_gauss = gauss.solve_linear(A, b)
if dump_matrices:
	print("gauss: x =", x_gauss)
print("gauss: ||Ax - b|| =", math_utils.vec_norm_3(np.matmul(A, x_gauss) - b))

x_seidel = seidel.solve_linear(A, b, 1e-15)
if dump_matrices:
	print("seidel: x =", x_seidel)
print("seidel: ||Ax - b|| =", math_utils.vec_norm_3(np.matmul(A, x_seidel) - b))