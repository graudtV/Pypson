# simle unit tests

import gauss
import seidel
import math_utils
import numpy as np

A = np.array([
	[1.0, 2, -2, 3],
	[3, 2, -1, 4],
	[5, 1, 2, -5],
	[2, -8, 3, 4]
	])

x = np.array([1, -4, 7,  -2])

# Ax = b
b = np.matmul(A, x)

def is_equal(vec1, vec2, precision):
	for x, y in zip(vec1, vec2):
		if abs(x - y) > precision:
			return False
	return True	

def assert_equal(vec1, vec2, precision):
	if not is_equal(vec1, vec2, precision):
		raise RuntimeError("vectors are not equal: {} != {}".format(vec1, vec2))

def run_unit_tests():
	assert_equal(gauss.solve_linear(A, b), x, 0.001)
	assert math_utils.vec_norm_1(np.array([-1, 2, -3, -7, 2]))== 7
	assert math_utils.vec_norm_2(np.array([-1, 2, -3, -7, 2])) == 15
	assert math_utils.vec_norm_3(np.array([-3, 4])) == 5
	assert math_utils.mrx_norm_1(np.array([[10, -2], [3, -4]])) == 12
	assert math_utils.minabs_eigen_value(np.array([[3, 0], [0, -4]])) == 3
	assert math_utils.minabs_eigen_value(np.array([[-3, 0], [0, -4]])) == -3
	assert math_utils.maxabs_eigen_value(np.array([[3, 0], [0, -4]])) == -4

	print("All tests passed")

if __name__ == "__main__":
	run_unit_tests()