import theano
import theano.tensor as T
import numpy as np

# define shared variables
k = T.iscalar("k")
n_sym = T.iscalar("n_sym")

results, updates = theano.scan(lambda: {k: (k + 1)}, n_steps=n_sym)
accumulator = theano.function([n_sym], [], updates=updates, allow_input_downcast=True)

print k.get_value()
accumulator(5)
print k.get_value()

# results, updates = theano.scan(lambda k: k + 1, outputs_info=k, n_steps=n_sym)
# accumulator = theano.function([k, n_sym], results[-1])
#
# # print k.get_value()
# aa = accumulator(0, 5)
# print aa
