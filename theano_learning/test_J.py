import theano
import theano.tensor as T
import numpy
from time import time
from theano.printing import pydotprint

# x = T.dvector('x')
# y = x ** 2
# J, updates = theano.scan(lambda i, y, x: T.grad(
#     y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y, x])
# f = theano.function([x], J)
# f_r = theano.function([x], y)
# print f_r([4, 4, 5])
# print f([4, 4, 5])
# print updates[i]

# W = T.dmatrix('W')
# V = T.dmatrix('V')
# x = T.dvector('x')
# y = T.dot(x, W)
# JV = T.Rop(y, W, V)
# f = theano.function([W, V, x], JV)
# print f([[1, 1], [1, 1]], [[2, 2], [2, 2]], [0, 1])

# W = T.dmatrix('W')
# v = T.dvector('v')
# x = T.dvector('x')
# y = T.dot(x, W)
# VJ = T.Lop(y, W, v)
# f = theano.function([v, x], VJ)
# print f([2, 2], [0, 1])

# coefficients = T.vector("coefficients")
# x = T.scalar("x")
#
# max_coefficients_supported = 10000
#
# # Generate the components of the polynomial
# components, updates = theano.scan(fn=lambda coefficient, power, free_variable: coefficient * (free_variable ** power),
#                                   outputs_info=None,
#                                   sequences=[coefficients, T.arange(max_coefficients_supported)],
#                                   non_sequences=x)
# # Sum them up
# polynomial = components.sum()
#
# # Compile a function
# calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=polynomial)
#
# # Test
# test_coefficients = numpy.asarray([1, 0, 2], dtype=numpy.float32)
# test_value = 3
# start_t = time()
# print(calculate_polynomial(test_coefficients, test_value))
# print time() - start_t
# start_t = time()
# print(1.0 * (3 ** 0) + 0.0 * (3 ** 1) + 2.0 * (3 ** 2))
# print time() - start_t

a = theano.shared(1)
values, updates = theano.scan(lambda: {a: a + 1}, n_steps=10)
b = a + 1
c = updates[a] + 1
f = theano.function([], [b, c], updates=updates)
f()
print(b)
print(c)
print(a.get_value())
pydotprint(b, outfile="b.png", var_with_name_simple=True)
pydotprint(f, outfile="f.png", var_with_name_simple=True)
