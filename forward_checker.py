import numpy as np
from rnn import BidirectionalRNN
from rnn import VanillaRNN
from optimizer import SGD
from optimizer import BFGS
from meta_rnn import MetaRNN
import theano.tensor as T
import theano
import logging

def gibbs_sampler_random(x, n0, n_steps, iterations, brnn):
    """
    :param x input data.
    :param n_steps number of steps to reconstruct.
    :param 
    """
    print n0
    for _ in xrange(iterations):
        for idx in xrange(n_steps):
            y = brnn.predict(x)
            x[idx+n0] = y[idx+n0]

    return x


def gibbs_sampler(x, n_steps, iterations, rnn, brnn):
    """
    :param x input data.
    :param n_steps number of steps to reconstruct.
    :param 
    """
    n0 = x.shape[0]
    y = rnn.sample(x, n_steps).reshape(n_steps, 1, -1)

    complete_sequence = np.vstack((x,y))
    for _ in xrange(iterations):
        for idx in xrange(n_steps):
            y = brnn.predict(complete_sequence)
            complete_sequence[idx+n0] = y[idx+n0]

    return complete_sequence
    

logging.basicConfig(level=logging.INFO)

seed = 0
prng = np.random.RandomState(seed)
n_in = 1
n_hidden = [2]
n_out = 1
activation = ['linear', 'linear']
bias_init = [0, 0]

brnn = BidirectionalRNN(n_in, n_hidden, n_out, activation, bias_init, prng)
rnn = VanillaRNN(n_in, n_hidden, n_out, activation, bias_init, prng)
opt = BFGS()
epochs = 200
L1_reg = 0
L2_reg = 0
burned_in = 0
loss_name = 'mse'
batch_size = 1
meta_brnn = MetaRNN(brnn, opt, epochs, batch_size, L1_reg, L2_reg, burned_in,
                                                       loss_name, mode=theano.Mode(linker='cvm'))
meta_rnn = MetaRNN(rnn, opt, epochs, batch_size, L1_reg, L2_reg, burned_in,
                                                       loss_name, mode=theano.Mode(linker='cvm'))

x_data = np.empty((6,3,1))
x_data[:,0,0] = np.arange(6)
x_data[:,1,0] = np.arange(6)
x_data[:,2,0] = np.arange(6)

meta_brnn.fit(x_data[:-1,:,:], x_data[1:,:,:])
meta_rnn.fit(x_data[:-1,:,:], x_data[1:,:,:])
n_steps = 10
y =  rnn.sample(x_data[0,0,:].reshape(-1,1,1), n_steps)
print "complete_sequence"
print y
iterations = 10

complete_sequence = np.vstack((x_data[0,0,:].reshape(-1,1,1), np.random.randn(n_steps).reshape(n_steps,1,1)))

y_10 = gibbs_sampler(x_data[0,0,:].reshape(-1,1,1), n_steps, iterations, rnn, brnn)
y_10_random = gibbs_sampler_random(complete_sequence, 0, n_steps, iterations, brnn)


iterations = 100
y_100 = gibbs_sampler(x_data[0,0,:].reshape(-1,1,1), n_steps, iterations, rnn, brnn)
y_100_random = gibbs_sampler_random(complete_sequence, 0, n_steps, iterations, brnn)

print y_10
print "random"
print y_10_random
print "------"
print y_100
print "random"
print y_100_random

print rnn.theta.get_value()




#y_hat_brnn = brnn.predict(x_data)
#y_hat_rnn = rnn.predict(x_data)

#print "--- y ---"
#print y_hat_brnn
#print y_hat_rnn
#print "--- hb ---"
#print hb
#print "--- xb ---"#
#print xb

