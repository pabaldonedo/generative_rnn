import numpy as np
from rnn import BidirectionalRNN
from rnn import HiddenRNN

import theano.tensor as T
import theano

def shared_dataset(self, data_xy, borrow=True):
    """ Load the dataset into shared variables """

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                                        borrow=True)

    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                                        borrow=True)

    return shared_x, shared_y

seed = 0
prng = np.random.RandomState(seed)
n_in = 1
n_hidden = [2]
n_out = 1
activation = ['linear', 'linear']
bias_init = [0, 0]

rnn = BidirectionalRNN(n_in, n_hidden, n_out, activation, bias_init, prng)

#Input and output variables with dimension (time, nseq, data dimensionality)
x = T.tensor3(name='x')
y = T.tensor3(name='y', dtype=theano.config.floatX)

#Connect rnn
rnn.define_network(x)
predict = theano.function(inputs=[x,], outputs=[rnn.y_t, rnn.forward_rnn.h, rnn.backward_rnn.h])




x_data = np.empty((5,3,1))
x_data[:,0,0] = np.arange(5)
x_data[:,1,0] = np.arange(5)
x_data[:,2,0] = np.arange(5)


y_hat, hf, hb = predict(x_data)
print y_hat.shape
print "--- y ---"
print y_hat
print "--- hf ---"
print hf
print "--- hb ---"

print hb


