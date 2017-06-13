import theano
import theano.tensor as T
import numpy as np

# define tensor variables
v = T.vector()
A = T.matrix()
y = T.tanh(T.dot(v, A))
results, updates = theano.scan(lambda i: T.grad(y[i], v), sequences=[T.arange(y.shape[0])])
compute_jac_t = theano.function([A, v], results, allow_input_downcast=True)  # shape (d_out, d_in)
abc = theano.function([A, v], y)

# test values
x = np.eye(5, dtype=theano.config.floatX)[0]
print(x.shape)
w = np.eye(5, 3, dtype=theano.config.floatX)
w[2] = np.ones((3), dtype=theano.config.floatX)
print(w.shape)
print(abc(w, x))
print(compute_jac_t(w, x))

# compare with numpy
print(((1 - np.tanh(x.dot(w)) ** 2) * w).T)
