import types
from types import IntType
from types import ListType
import numpy as np
import theano
import theano.tensor as T
import logging
from util import parse_activations


class RNN():
    def define_network(input):
        """To be implemented in subclasses. Sets up the network variables and connections."""
        self.y_pred = None
        raise NotImplementedError
class BidirectionalRNN(RNN):

    def __init__(self, n_in, n_hidden, n_out, activation, bias_init, prng):


        self.prng = prng

        self.n_in = n_in
        assert type(self.n_in) is IntType, "n_in must be an integer: {0!r}".format(self.n_in)

        assert type(n_hidden) is ListType, "n_hidden must be a list: {0!r}".format(n_hidden)
        self.n_hidden = np.array(n_hidden)

        self.activation_list = activation
        assert type(self.activation_list) is ListType, "activation must be a list:\
                                                                {0!r}".format(self.activation_list)

        assert len(self.n_hidden) + 1 == len(self.activation_list),\
        "Activation list must have len(n_hidden) + 1 values. Activation: {0!r}, n_hidden: \
                                                {1!r}".format(self.activation_list, self.n_hidden)

        self.bias_init = bias_init
        assert type(self.bias_init) is ListType, "biases initilizaition must be a list: \
                                                                     {0!r}".format(self.bias_init)

        assert len(self.bias_init) == len(n_hidden) + 1,\
        "Bias initialization list must have len(n_hidden) + 1 values. Bias list: {0!r}, n_hidden: \
                                                        {1!r}".format(self.bias_init, self.n_hidden)


        self.activation = parse_activations(self.activation_list)


        self.n_out = n_out
        assert type(self.n_out) is IntType, "n_out must be an int: {0!r}".format(self.n_out)

        self.forward_rnn = HiddenRNN(n_in, n_hidden, activation[:-1], bias_init[:-1], prng)
        self.backward_rnn = HiddenRNN(n_in, n_hidden, activation[:-1], bias_init[:-1], prng)
        self.type = 'BidirectionalRNN'
        self.opt = {'type': self.type, 'n_in': self.n_in, 'n_hidden': self.n_hidden,
                    'n_out': self.n_out, 'activation': self.activation_list,
                    'bias_init': self.bias_init}
        self.initialize_weights()
        self.complete_defined = False
        self.reconstruction_defined = False
        self.define_complete_network()
        self.define_reconstruction()


    def define_complete_network(self):
        """Sets connections for predicting all values given all inputs"""

        #self.wout_f_vector = theano.shared(value=np.zeros(theta_shape, dtype=theano.config.floatX))
        #self.wout_b_vector = theano.shared(value=np.zeros(theta_shape, dtype=theano.config.floatX))
        #self.out_bias = theano.shared(value=np.zeros(self.n_out, dtype=theano.config.floatX))

        #self.initialize_wout(self.wout_f, self.wout_b_vector)
        #self.initialize_wout(self.wout_f)
        #self.initialize_bias(self.out_bias)

        def step(htm1_f, htm1_b):
            y_t = self.activation[-1](T.dot(htm1_f, self.W_out_f) + T.dot(htm1_b, self.W_out_b) +
                                                                                    self.b)
            return y_t


        padding_f = T.alloc(0, 1, self.forward_rnn.h.shape[1], self.forward_rnn.h.shape[2])
        padding_b = T.alloc(0, 1, self.backward_rnn.h.shape[1], self.backward_rnn.h.shape[2])

        self.y_t, _ = theano.scan(step,
                    sequences=[T.concatenate([padding_f, self.forward_rnn.h[:-1]], axis=0), T.concatenate([self.backward_rnn.h[-2::-1], padding_b], axis=0)],
                    outputs_info=None)


        self.L1 = abs(self.W_out_f.sum()) + abs(self.W_out_b.sum()) + \
                                                        self.forward_rnn.L1 + self.backward_rnn.L1

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.W_out_f ** 2).sum() + (self.W_out_b ** 2).sum() +  \
                                                self.forward_rnn.L2_sqr + self.backward_rnn.L2_sqr

        self.predict_complete = theano.function(
                inputs=[self.x], outputs=self.y_t,
                givens={self.forward_rnn.x: self.x,
                        self.backward_rnn.x: self.x[::-1]})

        self.complete_defined = True

    def define_reconstruction(self):
        self.y_reconstruction = self.activation[-1](T.dot(self.forward_rnn.h[-1], self.W_out_f) +
                                T.dot(self.backward_rnn.h[-1], self.W_out_b) + self.b)

        self.mask = T.lscalar('mask')
        self.predict_reconstruction = theano.function(
                inputs=[self.x, self.mask], outputs=self.y_reconstruction,
                givens={self.forward_rnn.x: self.x[:self.mask],
                        self.backward_rnn.x: self.x[-1:self.mask:-1]})
        self.reconstruction_defined = True


    def initialize_weights(self):

        self.x = T.tensor3(name='x')
        hidden_weights_shape = np.sum(self.n_hidden ** 2) + self.n_in * self.n_hidden[0] + \
                    np.sum(self.n_hidden[:-1]*self.n_hidden[1:]) + \
                    np.sum(self.n_hidden) + np.sum(self.n_hidden)

        theta_shape = self.n_out*self.n_hidden[-1]*2+self.n_out + 2*hidden_weights_shape
        self.theta = theano.shared(value=np.zeros(theta_shape, dtype=theano.config.floatX))

        param_idx = 0
        Wr_init_forward, W_forward_init_forward, h0_init_forward, bh_init_forward = \
        self.forward_rnn.initialize_weights(self.theta[param_idx:(param_idx+hidden_weights_shape)])
        self.forward_rnn.define_network()

        param_idx += hidden_weights_shape
        Wr_init_backward, W_forward_init_backward, h0_init_backward, bh_init_backward = \
        self.backward_rnn.initialize_weights(self.theta[param_idx:(param_idx+hidden_weights_shape)])

        self.backward_rnn.define_network()
        param_idx += hidden_weights_shape

        param_idx, W_out_f_init, W_out_b_init = self.initialize_wout(param_idx)
        param_idx, b_out_init = self.initialize_out_bias(param_idx)

        assert param_idx == theta_shape

        self.pack_weights()

        self.theta.set_value(np.concatenate([x.ravel() for x in
                    (Wr_init_forward, W_forward_init_forward, h0_init_forward, bh_init_forward,
                    Wr_init_backward, W_forward_init_backward, h0_init_backward, bh_init_backward,
                    W_out_f_init, W_out_b_init, b_out_init)]))


    def pack_weights(self):
        """Packs weights and biases for convenience"""
        self.params = []
        self.params.append(self.W_out_f)
        self.params.append(self.W_out_b)
        self.params.append(self.b)


    def initialize_wout(self, param_idx):
        #W_out_f_init = np.asarray(self.prng.uniform(size=(self.n_hidden[-1], self.n_out),
        #                                          low=-0.01, high=0.01),
        #                                          dtype=theano.config.floatX)
        W_out_f_init = np.ones((self.n_hidden[-1], self.n_out), dtype=theano.config.floatX)

        self.W_out_f = self.theta[param_idx:(param_idx+self.n_hidden[-1]*self.n_out)].reshape(
                                                                    (self.n_hidden[-1], self.n_out))

        self.W_out_f.name = 'W_out_f'
        param_idx += self.n_hidden[-1]*self.n_out

        #W_out_b_init = np.asarray(self.prng.uniform(size=(self.n_hidden[-1], self.n_out),
        #                                          low=-0.01, high=0.01),
        #                                          dtype=theano.config.floatX)
        W_out_b_init = np.ones((self.n_hidden[-1], self.n_out), dtype=theano.config.floatX)

        self.W_out_b = self.theta[param_idx:(param_idx+self.n_hidden[-1]*self.n_out)].reshape(
                                                                    (self.n_hidden[-1], self.n_out))

        self.W_out_b.name = 'W_out_b'
        param_idx += self.n_hidden[-1]*self.n_out
        return param_idx, W_out_f_init, W_out_b_init


    def initialize_out_bias(self, param_idx):
        b_init = np.ones(self.n_out)*self.bias_init[-1]
        self.b = self.theta[param_idx:(param_idx+self.n_out)]
        param_idx += self.n_out
        return param_idx, b_init


class HiddenRNN(RNN):

    def __init__(self, n_in, n_hidden, activation, bias_init, prng):
        """Defines the basics of a Vanilla Recurrent Neural Network.

        :param n_in: integer defining the number of input units.
        :param n_hidden: list of integers defining the number of hidden units per layer.
        :param activation: list of size len(n_hidden) + 1 defining the activation function per layer.
        :param bias_init: list with bias initalization for [layers biases] + [output bias].
        :param prng: random number generator.
        """
        self.prng = prng

        self.n_in = n_in
        assert type(self.n_in) is IntType, "n_in must be an integer: {0!r}".format(self.n_in)

        assert type(n_hidden) is ListType, "n_hidden must be a list: {0!r}".format(n_hidden)
        self.n_hidden = np.array(n_hidden)

        self.activation_list = activation
        assert type(self.activation_list) is ListType, "activation must be a list:\
                                                                {0!r}".format(self.activation_list)

        assert len(self.n_hidden) == len(self.activation_list),\
        "Activation list must have len(n_hidden) + 1 values. Activation: {0!r}, n_hidden: \
                                                {1!r}".format(self.activation_list, self.n_hidden)

        self.bias_init = bias_init
        assert type(self.bias_init) is ListType, "biases initilizaition must be a list: \
                                                                     {0!r}".format(self.bias_init)

        assert len(self.bias_init) == len(n_hidden),\
        "Bias initialization list must have len(n_hidden) + 1 values. Bias list: {0!r}, n_hidden: \
                                                        {1!r}".format(self.bias_init, self.n_hidden)


        self.activation = parse_activations(self.activation_list)
        self.type = 'HiddenRNN'
        self.opt = {'type': self.type, 'n_in': self.n_in, 'n_hidden': self.n_hidden,
                    'activation': self.activation_list,
                    'bias_init': self.bias_init}
        self.defined = False


        logging.info('RNN loaded. Type: {0}, input layer: {1}, hidden layers: {2}'
            'activation: {3}'.format(self.type, self.n_in, self.n_hidden, self.activation))


    def define_network(self):
        """Sets all connections"""

        # recurrent function (using tanh activation function) and arbitrary output
        # activation function
        def step(x_t, h_tm1):

            ha_t = T.zeros_like(h_tm1)#T.alloc(np.zeros(np.sum(self.n_hidden)), h_tm1.shape[0], h_tm1.shape[1])
            h_t = T.zeros_like(h_tm1)#T.alloc(np.zeros(np.sum(self.n_hidden)), h_tm1.shape[0], h_tm1.shape[1])

            idx = 0
            for i, h_units in enumerate(self.n_hidden):
                if i == 0:
                    ha_t = T.set_subtensor(ha_t[:,idx:(idx+h_units)], T.dot(h_tm1[:,idx:(idx + h_units)], self.Wr[i]) + T.dot(x_t, self.W_forward[i]) + self.bh[i])
                    h_t = T.set_subtensor(h_t[:,idx:(idx+h_units)], self.activation[i](ha_t[:, idx:(idx+h_units)]))
                else:
                    ha_t = T.set_subtensor(ha_t[:,idx:(idx+h_units)], T.dot(h_tm1[:,idx:(idx + h_units)], self.Wr[i]) + T.dot(h_t[:,(idx-self.n_hidden[i-1]):idx], self.W_forward[i]) + self.bh[i])
                    h_t = T.set_subtensor(h_t[:,idx:(idx+h_units)], self.activation[i](ha_t[:, idx:(idx+h_units)]))

                idx += h_units

            return h_t

        # the hidden state `h` for the entire sequence, and the output for the
        # entire sequence `y` (first dimension is always time)
        # Note the implementation of weight-sharing h0 across variable-size
        # batches using T.ones multiplying h0
        # Alternatively, T.alloc approach is more robust

        self.h, _ = theano.scan(step,
                    sequences=self.x,
                    outputs_info=T.alloc(self.h0_as_vector, self.x.shape[1], np.sum(self.n_hidden)))

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        for w in self.Wr:
            self.L1 += abs(w.sum())

        for w in self.W_forward:
            self.L1 += abs(w.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        for w in self.Wr:
            self.L2_sqr += (w ** 2).sum()

        for w in self.W_forward:
            self.L2_sqr += (w ** 2).sum()

        self.defined = True

    def initialize_recursive_weights(self, param_idx):
        self.Wr = []
        Wr_init = np.empty(0)
        for i, h_units in enumerate(self.n_hidden):
            # recurrent weights as a shared variable
            self.Wr.append(self.theta[param_idx:(param_idx + h_units ** 2)].reshape(
                (h_units, h_units)))
            self.Wr[i].name = 'Wr_{0}'.format(i)
            #Wr_init = np.append(Wr_init, np.asarray(self.prng.uniform(size=(h_units, h_units),
            #                                      low=-0.01, high=0.01),
            #                                      dtype=theano.config.floatX).flatten())
            Wr_init = np.append(Wr_init, np.eye(h_units, dtype=theano.config.floatX).flatten())
            param_idx += h_units ** 2
        return param_idx, Wr_init

    def initialize_forward_weights(self, param_idx):
        self.W_forward = []
        W_forward_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):
            if i == 0:
                self.W_forward.append(self.theta[param_idx:(param_idx + self.n_in * \
                                          h_units)].reshape((self.n_in, h_units)))
                #W_forward_init = np.append(W_forward_init,
                #                                np.asarray(self.prng.uniform(size=(self.n_in, h_units),
                #                                low=-0.01, high=0.01),
                #                                dtype=theano.config.floatX))
                W_forward_init = np.append(W_forward_init, np.ones((self.n_in, h_units), dtype=theano.config.floatX))
                param_idx += self.n_in * self.n_hidden[0]

            else:
                self.W_forward.append(self.theta[param_idx:(param_idx + self.n_hidden[i-1] * \
                                          h_units)].reshape((self.n_hidden[i-1], h_units)))
        
                #W_forward_init = np.append(W_forward_init,
                #                        np.asarray(self.prng.uniform(size=(self.n_hidden[i-1], h_units),
                #                                    low=-0.01, high=0.01),
                #                                    dtype=theano.config.floatX))
    
                W_forward_init = np.append(W_forward_init, np.ones((self.n_hidden[i-1], h_units), dtype=theano.config.floatX))
                param_idx += self.n_hidden[i-1] * h_units

            self.W_forward[i].name = 'W_forward_{0}'.format(i)
        return param_idx, W_forward_init


    def initialize_hidden_units(self, param_idx):     
        h0_param_idx = param_idx

        self.h0 = []
        h0_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):

            self.h0.append(self.theta[param_idx:(param_idx + h_units)])
            self.h0[i].name = 'h0_{0}'.format(i)
            h0_init = np.append(h0_init, np.zeros((h_units,), dtype=theano.config.floatX))
            param_idx += h_units

        self.h0_as_vector = self.theta[h0_param_idx:param_idx]
        self.h0_as_vector.name = 'h0_as_vector'
        return param_idx, h0_init

    def initialize_bias(self, param_idx):

        self.bh = []
        bh_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):

            self.bh.append(self.theta[param_idx:(param_idx + h_units)])
            self.bh[i].name = 'bh_{0}'.format(i)
            bh_init = np.append(bh_init, self.bias_init[i]*np.ones((h_units,),
                                                                        dtype=theano.config.floatX))
            param_idx += h_units

        bh_init = np.array(bh_init)

        return param_idx, bh_init

    def initialize_weights(self, theta):
        """Returns Initialization values of shared variable theta and assigns it to the network weights"""
        

        self.x = T.tensor3(name='x')

        # theta is a vector of all trainable parameters
        # it represents the value of W, W_in, W_out, h0, bh, by
        
        self.theta = theta

        #Parameters are reshaped views of theta
        param_idx = 0  # pointer to somewhere along parameter vector
        param_idx, Wr_init = self.initialize_recursive_weights(param_idx)
        param_idx, W_forward_init = self.initialize_forward_weights(param_idx)
        param_idx, h0_init = self.initialize_hidden_units(param_idx)
        param_idx, bh_init = self.initialize_bias(param_idx)

        assert(param_idx == theta.shape.eval())

        self.pack_weights()
        return Wr_init, W_forward_init, h0_init, bh_init



    def pack_weights(self):
        """Packs weights and biases for convenience"""
        self.params = []

        for w in self.Wr:
            self.params.append(w)

        for w in self.W_forward:
            self.params.append(w)


        for h0 in self.h0:
            self.params.append(h0)

        for bh in self.bh:
            self.params.append(bh)


