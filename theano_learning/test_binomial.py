import theano
import theano.tensor as T
import numpy as np
from theano.printing import pydotprint

# define tensor variables
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

# define shared random stream
trng = T.shared_randomstreams.RandomStreams(1234)
d = trng.binomial(size=W[1].shape)

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym) * d, sequences=X)
compute_with_bnoise = theano.function(inputs=[X, W, b_sym], outputs=results,
                                      updates=updates, allow_input_downcast=True)
x = np.eye(10, 2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)

print(x)
print(updates)
print(compute_with_bnoise(x, w, b))
pydotprint(results, outfile="results.png", var_with_name_simple=True)

pydotprint(compute_with_bnoise, outfile="compute_with_bnoise.png", var_with_name_simple=True)
