# Gradient
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np
from random import random
from theano import pp
from theano.printing import pydotprint

# Define variables:
x = T.matrix('x')
w = theano.shared(np.array([random(), random()]))
b = theano.shared(1.)
learning_rate = 0.01

# Define mathematical expression:
z = T.dot(x, w) + b
a = 1 / (1 + T.exp(-z))

a_hat = T.vector('a_hat')  # Actual output
cost = -(a_hat * T.log(a) + (1 - a_hat) * T.log(1 - a)).sum()

dw, db = T.grad(cost, [w, b])
# print pp(db)
# f = theano.function([x, a_hat], db)
# print pp(f.maker.fgraph.outputs[0])

a1 = theano.shared(1)
values, updates = theano.scan(lambda: {a1: a1 + 1}, n_steps=10)
b1 = a1 + 1
c = updates[a1] + 1
f = theano.function([], [b1, c], updates=updates)
print(b1.get_value())
print(c.get_value())
print(a1.get_value())

train = theano.function(
    inputs=[x, a_hat],
    outputs=[a, cost],
    updates=[
        [w, w - learning_rate * dw],
        [b, b - learning_rate * db]
    ]
)

# Define inputs and weights
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
outputs = [0, 0, 0, 1]

# Iterate through all inputs and find outputs:
cost = []
for iteration in range(30000):
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter)

# Print the outputs:
print 'The outputs of the NN are:'
for i in range(len(inputs)):
    print 'The output for x1=%d | x2=%d is %.2f' % (inputs[i][0], inputs[i][1], pred[i])

# Plot the flow of cost:
# print '\nThe flow of cost during model run is as following:'
# import matplotlib.pyplot as plt
# plt.plot(cost)
# plt.show()
