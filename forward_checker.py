import numpy as np
from rnn import BidirectionalRNN
from rnn import HiddenRNN
from optimizer import SGD
from optimizer import BFGS
from meta_rnn import MetaRNN
import theano.tensor as T
import theano
import logging

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

logging.basicConfig(level=logging.INFO)

seed = 0
prng = np.random.RandomState(seed)
n_in = 1
n_hidden = [2]
n_out = 1
activation = ['linear', 'linear']
bias_init = [0, 0]

rnn = BidirectionalRNN(n_in, n_hidden, n_out, activation, bias_init, prng)
opt = SGD('constant', [1e-3], 'nesterov', 0.0001, mode=theano.Mode(linker='cvm'))
#opt = BFGS()
epochs = 200
L1_reg = 0
L2_reg = 0
burned_in = 0
loss_name = 'mse'
batch_size = 1
meta_rnn = MetaRNN(rnn, opt, epochs, batch_size, L1_reg, L2_reg, burned_in,
                                                       loss_name, mode=theano.Mode(linker='cvm'))



import theano.tensor as T

#y = T.tensor3(name='y', dtype=theano.config.floatX)
#y_pred = rnn.y_t
#loss = T.mean((y_pred - y) ** 2)

#gtheta = T.grad(loss, rnn.theta)



#th = theano.function(
#                inputs=[rnn.x, y], outputs=[rnn.theta, gtheta])


x_data = np.empty((5,3,1))
x_data[:,0,0] = np.arange(5)
x_data[:,1,0] = np.arange(5)
x_data[:,2,0] = np.arange(5)
#mask = 2
#print th(x_data, x_data)

#print x_data[mask,:,:].reshape(-1,x_data.shape[1], x_data.shape[2]).shape
meta_rnn.fit(x_data, x_data)

lr = 0.00001
#upd = rnn.theta - lr*gtheta
#gx = theano.function(
#                inputs=[rnn.x, rnn.mask], outputs=loss,
#                givens={rnn.forward_rnn.x: rnn.x[:rnn.mask],
#                        rnn.backward_rnn.x: rnn.x[-1:rnn.mask:-1],
#                        y: x_data[mask,:,:].reshape(-1,x_data.shape[1], x_data.shape[2]),},
 #               updates={rnn.theta: upd})
#t = []
#for i in xrange(100):
#    t.append(gx(x_data, mask))
#    if (i+1) % 10:
#        print i+1




#print t
#print th()
#print 
#meta_rnn.fit(x_data, x_data[mask,:,:], mask)
#Input and output variables with dimension (time, nseq, data dimensionality)
#x = T.tensor3(name='x')
#y = T.tensor3(name='y', dtype=theano.config.floatX)

#Connect rnn
#rnn.define_network()
#predict = theano.function(inputs=[x,], outputs=[rnn.y_t, rnn.forward_rnn.h, rnn.backward_rnn.h])




#y_hat, hf, hb = predict(x_data)
y_hat = rnn.predict_complete(x_data)

print "--- y ---"
print y_hat
print np.mean((y_hat-x_data)**2)
#print "--- hf ---"
#print hf
#print "--- hb ---"
#print hb
#print "--- xb ---"#
#print xb

