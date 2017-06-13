import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

# Define variables:
x = T.vector('x')
w = theano.shared(np.array([1, 1]))
b = theano.shared(-1.5)

# Define mathematical expression:
z = T.dot(x, w) + b
a = ifelse(T.lt(z, 0), 0, 1)

neuron = theano.function([x], a)

# Define inputs and weights
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

# Iterate through all inputs and find outputs:
for i in range(len(inputs)):
    t = inputs[i]
    out = neuron(t)
    print 'The output for x1=%d | x2=%d is %d' % (t[0], t[1], out)
