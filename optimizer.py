import warnings
import types
import scipy
import theano
import theano.tensor as T
from types import FloatType
from types import IntType
from types import ListType
from types import TupleType
import numpy as np
import logging
from collections import OrderedDict


class Optimizer():

    def fiting_variables(self, batch_size, train_set_x, test_set_x=None):
        """Sets useful variables for locating batches"""    
        self.index = T.lscalar('index')    # index to a [mini]batch
        self.n_ex = T.lscalar('n_ex')      # total number of examples
        assert type(batch_size) is IntType or FloatType, "Batch size must be an integer."
        if type(batch_size) is FloatType:
            warnings.warn('Provided batch_size is FloatType, value has been truncated')
            batch_size = int(batch_size)
        # Proper implementation of variable-batch size evaluation
        # Note that the last batch may be a smaller size
        # So we keep around the effective_batch_size (whose last element may
        # be smaller than the rest)
        # And weight the reported error by the batch_size when we average
        # Also, by keeping batch_start and batch_stop as symbolic variables,
        # we make the theano function easier to read
        self.batch_start = self.index * batch_size
        self.batch_stop = T.minimum(self.n_ex, (self.index + 1) * batch_size)
        self.effective_batch_size = self.batch_stop - self.batch_start

        self.get_batch_size = theano.function(inputs=[self.index, self.n_ex],
                                          outputs=self.effective_batch_size)

        # compute number of minibatches for training
        # note that cases are the second dimension, not the first
        self.n_train = train_set_x.get_value(borrow=True).shape[1]
        self.n_train_batches = int(np.ceil(1.0 * self.n_train / batch_size))
        if test_set_x is not None:
            self.n_test = test_set_x.get_value(borrow=True).shape[1]
            self.n_test_batches = int(np.ceil(1.0 * self.n_test / batch_size))

    def fit(self):
        """To be implemented in subclasses. Performs the optimization."""
        raise NotImplementedError

class GradientBased(Optimizer):

    def get_updates(self, theta, cost=None, gtheta=None):
        raise NotImplementedError

    def fit(self, x, y, train_set_x, train_set_y, batch_size, cost, theta, n_epochs,
                            compute_error, call_back, test_set_x=None, test_set_y=None,
                            validate_every=1):
        """Performs the optimization using a Gradient Based algorithm.

        :param rnn: recurent neural network
        :param train_set_x: Theano shared variable set to training set matrix.
                            Dimensions: sequence length x #training samples x input dimensionality.
        :param train_set_y: Theano shared variable set to objective matrix.
                            Dimensions:sequence length x #training samples x output dimensionality.
        :param batch_size: integer #samples used per batch.
        :param cost: theano variable equals to the cost of the rnn.
        :param n_epochs: integer number of epochs to train.
        :param compute_error: theano function that computes error given predicted output and true output.
        :param call_back: call back function to call for printing information.
        :param test_set_x: test set matrix.
                            Dimensions: sequence length x #training samples x input dimensionality. 
        :param validate_every: telling every how many epochs the fit functions reports to the
                               call_back function. It can be a float.
        """

        self.test_availavility = test_set_x is not None
        #Setting up indicator variables for looping along batches

        self.fiting_variables(batch_size, train_set_x, test_set_x=test_set_x)

        gtheta = T.grad(cost, theta)

        #Variables to keep track of the error while training
        train_error_evolution = []
        test_error_evolution = []

        updates = self.get_updates(theta, gtheta=gtheta)

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`

        train_model = theano.function(inputs=[self.index, self.n_ex],
            outputs=cost,
            updates=updates,
            givens={x: train_set_x[:, self.batch_start:self.batch_stop],
                    y: train_set_y[:, self.batch_start:self.batch_stop]},
            mode=self.mode, on_unused_input='warn')

        epoch = 0
        #validate_every is specified in terms of epochs
        validation_frequency = np.round(validate_every * self.n_train_batches)

        while (epoch < n_epochs):
            epoch = epoch + 1

            for minibatch_idx in xrange(self.n_train_batches):

                minibatch_avg_cost = train_model(minibatch_idx, self.n_train)
                # iteration number (how many weight updates have we made?)
                # epoch is 1-based, index is 0 based
                iter = (epoch - 1) * self.n_train_batches + minibatch_idx + 1

                if iter % validation_frequency == 0:
                    this_train_loss  = compute_error(train_set_x.eval(), train_set_y.eval())

                    train_error_evolution.append((epoch, this_train_loss))
                    if self.test_availavility:
                        this_test_loss = compute_error(test_set_x.eval(), test_set_y.eval())
                        test_error_evolution.append((epoch, this_test_loss))

                        call_back(epoch, minibatch_idx + 1, self.n_train_batches,
                                        train_error=this_train_loss, opt_parameters=self.opt_parameters,
                                        test_error=this_test_loss)
                    else:
                        call_back(epoch, minibatch_idx + 1, self.n_train_batches,
                                        train_error=this_train_loss, opt_parameters=self.opt_parameters)
        
        if self.test_availavility:
            return train_error_evolution, test_error_evolution

        return train_error_evolution

class RMSProp(GradientBased):

    def __init__(self, learning_rate, rho, epsilon, mode=theano.Mode(linker='cvm')):
        self.learning_rate = learning_rate
        assert type(self.learning_rate) is FloatType or IntType, "Learning rate must be an integer or float: {0!r}".format(self.learning_rate)
        assert 0 < self.learning_rate, "Learning rate must be positive: {0!r}".format(self.learning_rate)
        self.rho = rho
        assert type(self.rho) is FloatType or IntType, "Rho decay must be an integer or float: {0!r}".format(self.rho)
        assert 0 < self.rho, "Rho decay must be positive: {0!r}".format(self.rho)
        self.epsilon = epsilon
        assert type(self.epsilon) is FloatType or IntType, "Epsilon must be an integer or float: {0!r}".format(self.epsilon)
        assert 0 < self.epsilon, "Epsilon must be positive: {0!r}".format(self.epsilon)  
        self.mode = mode
        self.opt_parameters = {'opt': 'RMSProp', 'lr':self.learning_rate, 'rho':self.rho,'e':self.epsilon}

        logging.info('Optimizer loaded. Type: {0}, learning rate: {1}, rho decay: {2},'
            ' epsilon: {3}'.format(self.opt_parameters['opt'], self.opt_parameters['lr'],
                self.opt_parameters['rho'], self.opt_parameters['e']))

    def get_updates(self, theta, cost=None, gtheta=None):
        if gtheta is None:
            assert cost is not None
            gtheta = T.grad(cost, theta)

        updates = []
        rms = theano.shared(theta.get_value() *0.)
        rms_upd = self.rho*rms + (1 - self.rho) * T.sqr(gtheta)
        theta_upd = theta - self.learning_rate * gtheta / T.sqrt(rms_upd + self.epsilon)

        updates.append((rms, rms_upd))
        updates.append((theta, theta_upd))
        return updates


class AdaDelta(GradientBased):

    def __init__(self, learning_rate, rho, epsilon, mode=theano.Mode(linker='cvm')):
        self.learning_rate = learning_rate
        assert type(self.learning_rate) is FloatType or IntType, "Learning rate must be an integer or float: {0!r}".format(self.learning_rate)
        assert 0 < self.learning_rate, "Learning rate must be positive: {0!r}".format(self.learning_rate)
        self.rho = rho
        assert type(self.rho) is FloatType or IntType, "Rho decay must be an integer or float: {0!r}".format(self.rho)
        assert 0 < self.rho, "Rho decay must be positive: {0!r}".format(self.rho)
        self.epsilon = epsilon
        assert type(self.epsilon) is FloatType or IntType, "Epsilon must be an integer or float: {0!r}".format(self.epsilon)
        assert 0 < self.epsilon, "Epsilon must be positive: {0!r}".format(self.epsilon)  
        self.mode = mode
        self.opt_parameters = {'opt': 'AdaDelta', 'lr':self.learning_rate, 'rho':self.rho,'e':self.epsilon}
        logging.info('Optimizer loaded. Type: {0}, learning rate: {1}, rho decay: {2},'
            ' epsilon: {3}'.format(self.opt_parameters['opt'], self.opt_parameters['lr'],
                self.opt_parameters['rho'], self.opt_parameters['e']))

    def get_updates(self, theta, cost=None, gtheta=None):
        if gtheta is None:
            assert cost is not None
            gtheta = T.grad(cost, theta)

        updates = []

        eg2 = theano.shared(theta.get_value() * 0.) 
        edth2 = theano.shared(theta.get_value() * 0.)

        eg2_upd = self.rho*eg2 + (1-self.rho)*T.sqr(gtheta)

        rms_dth_tm1 = T.sqrt(edth2 + self.epsilon)
        rms_gtheta_t = T.sqrt(eg2_upd + self.epsilon)
        dth = - gtheta * rms_dth_tm1/ rms_gtheta_t
        edth2_upd = self.rho*edth2 + (1-self.rho)*dth**2
        theta_upd = theta + self.learning_rate*dth

        updates.append((eg2, eg2_upd))
        updates.append((edth2, edth2_upd))
        updates.append((theta, theta_upd))
        return updates


class AdaGrad(GradientBased):

    def __init__(self, learning_rate, epsilon, mode=theano.Mode(linker='cvm')):
        self.learning_rate = learning_rate
        assert type(self.learning_rate) is FloatType or IntType, "Learning rate must be an integer or float: {0!r}".format(self.learning_rate)
        assert 0 < self.learning_rate, "Learning rate must be positive: {0!r}".format(self.learning_rate)
        self.epsilon = epsilon
        assert type(self.epsilon) is FloatType or IntType, "Epsilon must be an integer or float: {0!r}".format(self.epsilon)
        assert 0 < self.epsilon, "Epsilon must be positive: {0!r}".format(self.epsilon)  
        self.opt_parameters = {'opt': 'AdaGrad', 'learning_rate':self.learning_rate, 'epsilon':self.epsilon}
        self.mode = mode
        logging.info('Optimizer loaded. Type: {0}, learning rate: {1},'
            ' epsilon: {3}'.format(self.opt_parameters['opt'], self.opt_parameters['lr'],
                                    self.opt_parameters['e']))

    def get_updates(self, theta, cost=None, gtheta=None):
        """Return a list with tuples (variable, update expresion)
        :param cost: theano variable containing the cost to be minimized.
        :param theta: theano sahred variable with the weights to be optimized.
        :param gtheta: theano variable containing the gradients of theta. If None, the gradients are
        computed inside the function. Default: None.
        """
        updates = []
        if gtheta is None:
            assert cost is not None
            gtheta = T.grad(cost, theta)

        g2_sum = theano.shared(theta.get_value() * 0.)
        g2_sum_upd = g2_sum + T.sqr(gtheta) 
        theta_upd = theta - self.learning_rate * gtheta/T.sqrt(g2_sum_upd + self.epsilon) 
        
        updates.append((g2_sum, g2_sum_upd))
        updates.append((theta, theta_upd))
        return updates

class Adam(GradientBased):
    def __init__(self, step_size, b1, b2, e, mode=theano.Mode(linker='cvm')):
        """:param  step_size. positive float value.
        :param b1: Beta1 parameter in Adam. Float value in range [0-1).
        :param b2: Beta2 parameter in Adam. Float value in range [0-1).
        :param e: epsilon parameter. Float value.
        """
        self.step_size = step_size
        assert type(self.step_size) is FloatType or IntType, "Step size must be an integer or float: {0!r}".format(self.step_size)
        assert 0 < self.step_size, "Step size must be positive: {0!r}".format(self.step_size)
        self.b1 = b1
        assert type(self.b1) is FloatType or IntType, "B1 must be a float or integer: {0!r}".format(self.b1)
        assert 0 <= self.b1 < 1, "B1 must be in range [0, 1): {0!r}".format(self.b1)
        self.b2 = b2
        assert type(self.b2) is FloatType or IntType, "B2 must be a float or integer: {0!r}".format(self.b2)
        assert 0 <= self.b2 < 1, "B2 must be in range [0, 1): {0!r}".format(self.b2)
        self.e = e
        assert type(self.e) is FloatType or IntType, "Epsilon must be an integer or float: {0!r}".format(self.e)
        assert 0 < self.e, "Epsilon must be positive: {0!r}".format(self.e)  
        self.mode = mode
        self.opt_parameters = {'opt': 'Adam', 'alpha':self.step_size, 'b1':self.b1,
                                                                        'b2':self.b2, 'e':self.e}
        logging.info('Optimizer loaded. Type: {0}, step size: {1}, b1: {2},'
            ' b2: {3}, epsilon: {4}'.format(self.opt_parameters['opt'], self.opt_parameters['alpha'],
                self.opt_parameters['b1'], self.opt_parameters['b2'], self.opt_parameters['e']))

    def get_updates(self, theta, cost=None, gtheta=None):
        """Return a list with tuples (variable, update expresion)
        :param cost: theano variable containing the cost to be minimized.
        :param theta: theano sahred variable with the weights to be optimized.
        :param gtheta: theano variable containing the gradients of theta. If None, the gradients are
        computed inside the function. Default: None.
        """
        updates = []
        if gtheta is None:
            assert cost is not None
            gtheta = T.grad(cost, theta)
        i = theano.shared(np.asarray(0., dtype=theano.config.floatX))#floatX(0.))
        i_t = i + 1.
        fix1 = 1. - self.b1**i_t
        fix2 = 1. - self.b2**i_t
        lr_t = self.step_size * (T.sqrt(fix2) / fix1)

        m = theano.shared(theta.get_value() * 0.)
        v = theano.shared(theta.get_value() * 0.)
        m_t = self.b1 * m + (1-self.b1) * gtheta
        v_t = self.b2 * v + (1-self.b2) * T.sqr(gtheta)
        g_t = m_t / (T.sqrt(v_t) + self.e)
        theta_t = theta - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((theta, theta_t))
        updates.append((i, i_t))
        return updates


class SGD(GradientBased):

    def __init__(self, lr_decay_schedule, lr_decay_parameters, momentum_type, momentum, mode=theano.Mode(linker='cvm')):
        """
        :param lr_decay_schedule: type learning rate decay schedule used in learning_rate_scheduler.
        :param lr_decay_parameters: used and documented in learning_rate_scheduler.
        :param momentum_type: string either 'classic' or 'nesterov' defining the type of momentum.
        :param momentum: integer or float defining momentum.
        """
        self.lr_update = self.learning_rate_scheduler(lr_decay_schedule, lr_decay_parameters)
        self.mode = mode
        self.momentum_type = momentum_type
        self.momentum_types = ['classic', 'nesterov']
        assert self.momentum_type in momentum_type, "Momentum type not implemented: {0!r}. Try wiht one of the following: {1}".format(self.momentum_type, self.momentum_types)
        
        assert type(momentum) is FloatType or IntType, "Momentum parameter must be an int or float: {0!r}".format(momentum)
        assert momentum >=0, "Momentum must be >=0: {0!r}".format(momentum)
        self.momentum_update = lambda epoch: momentum #TODO momentum schedules
        logging.info('SGD optimizer object created')
        self.opt_parameters = {'opt': 'SGD', 'mom':momentum, 'momentum_type':self.momentum_type, 'lr_decay_schedule':lr_decay_schedule, 'lr_decay_parameters':lr_decay_parameters}
        
        logging.info('Optimizer loaded. Type: {0}, momentum_type: {1},'
            ' momentum: {2}, learning rate decay schedule: {3},'
            ' learning rate decay parameters([n(0), r, c]: {4})'.format(self.opt_parameters['opt'],
                self.opt_parameters['momentum_type'],
                self.opt_parameters['mom'], self.opt_parameters['lr_decay_schedule'],
                self.opt_parameters['lr_decay_parameters']))

    def learning_rate_scheduler(self, lr_decay_schedule, lr_decay_parameters):
        """
        Returns a function that gives the learning rate for this epoch taking as input the current epoch
        :param lr_decay_schedule: defines the learning decay schedule.
        Possible schedules:
            constant: n(t) = n(0)
            multiplication: n(t) = n(0)*r^t
            exponential: n(t) = n(0)* 10^(-t/r)
            power: n(t) = n(0)(1+t/t)^-c

        :param lr_decay_parameters: [n(0), r, c]
        """
        #Available learning rate decay schedules
        self.lr_decay_schedules = ['constant', 'multiplication', 'exponential', 'power']
        
        #Checks if asked schedule is implemented
        assert lr_decay_schedule in self.lr_decay_schedules, \
            "Learning rate schedule {0!r} not implemented. Choose one of the following: {1}".format(
                    lr_decay_schedule, self.lr_decay_schedules)

        assert type(lr_decay_parameters) is ListType or TupleType,\
        "Learning rate decay parameters must be a list or a tuple. Provided value: {0!r} and \
        type {1!r}".format(lr_decay_parameters, type(lr_decay_parameters))

        #Initial learning rate value for epoch 0.
        init_lr = lr_decay_parameters[0]

        assert type(init_lr) is FloatType or IntType,\
                        "Initial learning_rate value must be float or int: {0!r}".format(init_lr)

        assert init_lr >= 0, "Initial learning_rate must be >=0: {0!r}".format(init_lr)

        #Constant learning rate
        if lr_decay_schedule == self.lr_decay_schedules[0]:
            return lambda epoch: init_lr

        assert len(lr_decay_parameters) >= 2,\
        "If decay not constant at least two parameters [init learning_rate, decay rate] are\
        expected for learning_rate_decay_parameters. Provided values: {0!r}".format(lr_decay_parameters)

        #Parameter r
        decay_rate = lr_decay_parameters[1]

        assert type(decay_rate) is FloatType or IntType,\
                "Decay rate for learning rate value must be float or int: {0!r}".format(decay_rate)

        assert decay_rate >= 0, "Decay rate for learning rate must be >=0: {0!r}".format(decay_rate)

        if lr_decay_schedule == self.lr_decay_schedules[1]:
            return lambda epoch: init_lr*decay_rate**epoch

        if lr_decay_schedule == self.lr_decay_schedules[2]:
            return lambda epoch: init_lr*10**(-epoch*1./decay_rate)

        if lr_decay_schedule == self.lr_decay_schedules[3]:
            assert len(lr_decay_parameters) == 3, "For Power Scheduling 3 parameters are needed \
                                     [init learning rate, r, c]: {0!r}".format(lr_decay_parameters)

            c = lr_decay_parameters[2]
            assert type(c) is FloatType or IntType, \
                "c parameter in Power Schedule for lr must be float or int: {0!r}".format(type(c)) 

            assert c >=0, "c in Power Schedule for lr must be >=0: {0!r}".format(c)
            return lambda epoch: init_lr*(1+epoch*1./r)**-c

    def get_updates(self, theta, cost=None, gtheta=None):
        """Return a list with tuples (variable, update expresion)
        :param cost: theano variable containing the cost to be minimized.
        :param theta: theano sahred variable with the weights to be optimized.
        :param gtheta: theano variable containing the gradients of theta. If None, the gradients are
        computed inside the function. Default: None.
        """
        if gtheta is None:
            assert cost is not None
            #Gradient of weights
            gtheta = T.grad(cost, theta)

        i = theano.shared(np.asarray(0., dtype=theano.config.floatX))#floatX(0.))
        i_t = i + 1.

        l_r = self.lr_update(i_t)
        mom = self.momentum_update(i_t)  # momentum

        updates = []
        #Previous update needed for momentum. Initalized to 0.
        v = theano.shared(value=np.zeros(theta.size.eval(), dtype=theano.config.floatX))
        v_upd = mom * v - l_r * gtheta

        if self.momentum_type == self.momentum_types[0]:
            theta_new = theta + v_upd
        elif self.momentum_type == self.momentum_types[1]:
            theta_new = theta + mom * v_upd - l_r * gtheta
        else:
            raise NotImplementedError

        updates.append((v, v_upd))
        updates.append((theta, theta_new))
        self.opt_parameters['lr'] = l_r
        self.opt_parameters['mom'] = mom

        return updates


class ScipyBased(Optimizer):

    def train_fn(self, x_train, y_train, compute_error):
        """Returns the function used in the optimizer for evaluating the objective error and
        updates theta (RNN weights).
        :param x_train: training sample matrix. 
                            Dimensions: sequence length x #training samples x input dimensionality.
        :param y_train: training output matrix.
                            Dimensions: sequence length x #training samples x input dimensionality.
        :param compute_error: theano function that ouptuts the error given x and y.
        """
        def train(theta_value, theta):
            theta.set_value(theta_value, borrow=True)
            return compute_error(x_train, y_train)

        return train

    def train_fn_grad(self, theta_value, theta):
        """Returns the traing gradient used in the optimizer
        :param theta_value: vector of the new values of theta. Provided by scipy function.
        :param theta: theano shared variable of the rnn weights. Provided by scipy function in args.
        """
        theta.set_value(theta_value, borrow=True)

        train_grads = [self.batch_grad(i, self.n_train)
                        for i in xrange(self.n_train_batches)]
        train_batch_sizes = [self.get_batch_size(i, self.n_train)
                             for i in xrange(self.n_train_batches)]

        return np.average(train_grads, weights=train_batch_sizes,
                          axis=0)

    def get_optimizer_callback(self, x_train, y_train, compute_error, train_error_evolution, callback,
                                    opt_type, theta, validate_every, test_error_evolution=None,
                                    x_test=None, y_test=None):
        """Returns callback to be given to scipy optimizer.
        :param x_train: training sample matrix. 
                            Dimensions: sequence length x #training samples x input dimensionality.
        :param y_train: training output matrix.
                            Dimensions: sequence length x #training samples x input dimensionality.
        :param compute_error: theano function that ouptuts the error given x and y.
        :param train_error_evolution: list where to store the train error every validate_every epoch.
        :param callback: callback function in meta_rnn.
        :param opt_type: string to report which type of gradient based is being used. Ex: BFGS.
        :param theta: theano shared variable of the rnn weights. Provided by scipy function in args.
        :param validate_every: integer representing over how many epochs call_back is being called.
        :param test_error_evolution: list where to store the test error every validate_every epoch.
        :param x_test: test sample matrix. 
                            Dimensions: sequence length x #training samples x input dimensionality.
        :param y_test:: test output matrix.
                            Dimensions: sequence length x #training samples x input dimensionality.        """
        test_availavility = x_test is not None
        if test_availavility:
            assert y_test is not None
            assert test_error_evolution is not None
        self.epoch = 0

        def optimizer_callback(theta_value):
            self.epoch += 1
            if self.epoch % validate_every == 0:
                theta.set_value(theta_value, borrow=True)
                this_train_error = compute_error(x_train, y_train)
                train_error_evolution.append((self.epoch, this_train_error))
                if test_availavility:
                    this_test_error = compute_error(x_test, y_test)
                    test_error_evolution.append((self.epoch, this_test_error))
                    callback(self.epoch, self.n_train_batches, self.n_train_batches,
                        train_error=this_train_error, opt_parameters=self.opt_parameters,
                        test_error = this_test_error)
                else:
                    callback(self.epoch, self.n_train_batches, self.n_train_batches,
                        train_error=this_train_error, opt_parameters=self.opt_parameters)

        return optimizer_callback

    def fit(self, x, y, train_set_x, train_set_y, batch_size, cost, theta, n_epochs,
                            compute_error, call_back, test_set_x=None, test_set_y=None,
                            validate_every=1):

        """Performs the optimization using SGD.

        :param x: theano input variable of the rnn.
        :param y: theano output variable of the rnn.
        :param train_set_x: Theano shared variable set to training set matrix.
                            Dimensions: sequence length x #training samples x input dimensionality.
        :param train_set_y: Theano shared variable set to objective matrix.
                            Dimensions:sequence length x #training samples x output dimensionality.
        :param batch_size: integer #samples used per batch.
        :param cost: theano variable equals to the cost of the rnn.
        :param theta: theano shared variable containing all the weights (and biases) in the rnn.
        :param n_epochs: integer number of epochs to train.
        :param compute_train_error: theano function that computes train_error.
        :param call_back: call back function to call for printing information.
        :param compute_test_error: theano function that computes test_error.
        :param test_set_x: test set matrix.
                            Dimensions: sequence length x #training samples x input dimensionality. 
        :param validate_every: telling every how many epochs the fit functions reports to the
                               call_back function. It can be a float.
        """

        self.fiting_variables(batch_size, train_set_x, test_set_x=test_set_x)
        # compile a theano function that returns the gradient of the
        # minibatch with respect to theta
        self.batch_grad = theano.function(inputs=[self.index, self.n_ex],
                outputs=T.grad(cost, theta),
                givens={x: train_set_x[:, self.batch_start:self.batch_stop],
                        y: train_set_y[:, self.batch_start:self.batch_stop]},
                mode=self.mode, name="batch_grad")

        train_error_evolution = []
        if test_set_x is not None:
            test_error_evolution = []
            call_back_for_opt = self.get_optimizer_callback(train_set_x.eval(), train_set_y.eval(),
                                                compute_error, train_error_evolution,
                                                call_back, self.opt_parameters['opt'], theta,
                                                validate_every,
                                                test_error_evolution=test_error_evolution,
                                                x_test=test_set_x.eval(), y_test=test_set_y.eval())
        else:
            call_back_for_opt = self.get_optimizer_callback(train_set_x.eval(), train_set_y.eval(),
                                                        compute_error, train_error_evolution,
                                                        call_back, self.opt_parameters['opt'], theta,
                                                        validate_every)
              
        opt_train_function = self.train_fn(train_set_x.eval(), train_set_y.eval(), compute_error)

        best_theta = scipy.optimize.minimize(fun=opt_train_function,
                                    method=self.opt_parameters['opt'],
                                    x0=theta.get_value(),
                                    args=(theta,),
                                    jac=self.train_fn_grad,
                                    callback=call_back_for_opt,
                                    options={'disp':1,
                                    'retall':1,
                                    'maxiter':n_epochs,
                                    'gtol':self.gtol})
        if test_set_x is not None:
            return train_error_evolution, test_error_evolution
        return train_error_evolution


class BFGS(ScipyBased):

    def __init__(self, gtol=1e-5, mode=theano.Mode(linker='cvm')):
        self.mode = mode
        self.gtol = gtol
        self.opt_parameters = {'opt': 'bfgs', 'gtol':self.gtol}
        logging.info('Optimizer loaded. Type: {0}, gtol: {1}'.format(self.opt_parameters['opt'],
                                                                    self.opt_parameters['gtol']))


class CongugateGradient(ScipyBased):

    def __init__(self, gtol=1e-5, mode=theano.Mode(linker='cvm')):
        self.mode = mode
        self.gtol = gtol
        self.opt_parameters = {'opt': 'cg', 'gtol':self.gtol}
        logging.info('Optimizer loaded. Type: {0}, gtol: {1}'.format(self.opt_parameters['opt'],
                                                                    self.opt_parameters['gtol']))

        
class LBFGSB(ScipyBased):

    def __init__(self, gtol=1e-5, mode=theano.Mode(linker='cvm')):
        self.mode = mode
        self.gtol = gtol
        self.opt_parameters = {'opt': 'l-bfgs-b', 'gtol':self.gtol}
        logging.info('Optimizer loaded. Type: {0}, gtol: {1}'.format(self.opt_parameters['opt'],
                                                                    self.opt_parameters['gtol']))