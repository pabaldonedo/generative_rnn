from util import load_json
from util import define_loss
from types import IntType
from types import FloatType
import theano.tensor as T
import theano
import numpy as np
import logging


class MetaRNN():

    def __init__(self, rnn, optimizer, epochs, batch_size, L1_reg, L2_reg, burned_in,
                                                        loss_name, mode=theano.Mode(linker='cvm')):
        """Controller of the RNN problem
        :param rnn: instance of a subclass of rnn_class.RNN. It is the RNN that models the problem.
        :param optimizer: instance of a sublcass of optimizer.Optimizer. Optimizer to train RNN.
        :parm option_file: filename of json file containing training epochs, batch size, 
                            norm 1 regularizer weight, norm 2 regularizer weight,
                            burned in parameter and loss function to be used in training.
        """
        self.rnn = rnn
        self.optimizer = optimizer
        self.mode = mode

        self.epochs = epochs
        assert type(self.epochs) is IntType, "Provided epochs value is not an int: {0!r}".format(self.epochs)
        assert self.epochs > 0, "Provided epochs value is not valid: {0!r}".format(self.epochs)

        self.batch_size = batch_size
        assert type(self.batch_size) is IntType, "Provided batch size value is not an int: {0!r}".format(self.batch_size)
        assert self.batch_size > 0, "Provided batch size value is not valid: {0!r}".format(self.batch_size)

        self.L1_reg = L1_reg
        assert type(self.L1_reg) is IntType or FloatType, "Provided L1 regularization value is not an int or float: {0!r}".format(self.L1_reg)
        assert self.L1_reg >= 0, "Provided L1 regularization value is not valid: {0!r}".format(self.L1_reg) 

        self.L2_reg = L2_reg
        assert type(self.L2_reg) is IntType or FloatType, "Provided L2 regularization value is not an int or float: {0!r}".format(self.L2_reg)
        assert self.L2_reg >= 0, "Provided L2 regularization value is not valid: {0!r}".format(self.L2_reg) 

        self.burned_in = burned_in
        assert type(self.burned_in) is IntType, "Provided burned_in regularization value is not an int: {0!r}".format(self.burned_in)
        assert self.burned_in >= 0, "Provided burned_in regularization value is not valid: {0!r}".format(self.burned_in) 

        self.loss_name = loss_name
        self.loss = define_loss(self.loss_name)
        self.y = T.tensor3(name='y')

        logging.info('MetaRNN loaded. Epochs: {0}, batch_size: {1}, L1_reg: {2}, L2_reg: {3}, burned_in: {4}, loss: {5}'.format(self.epochs, self.batch_size, self.L1_reg, self.L2_reg, self.burned_in, self.loss_name))

    def opt_parameters_message(self, opt_parameters):
        msg = ''
        for key, value in opt_parameters.iteritems():
            msg += ', {0}: {1}'.format(key, value)
        return msg

    def get_call_back_function(self, x_train, y_train, x_test=None, y_test=None):
        test_available = x_test is not None
        if test_available:
            assert y_test is not None


        def epoch_display(epoch, minibatch, total_minibatch, train_error=None,
                                                                test_error=None, opt_parameters={}):
            """Call back function called by self.optimizer every validation_frequency epochs (default=1)"""


            if train_error is None:
                train_error = self.compute_error(x_train, y_train)

            if test_available:
                if test_error is None:
                    test_error = self.compute_error(x_test, y_test)

                if test_error < self.best_cost:
                    self.best_cost = test_error
                    self.best_rnn_fit = self.rnn.theta.get_value()

                logging.info('epoch {0}, minibatch {1}/{2}, tr loss {3}, test loss {4} best: {5}{6}'.format(
                        epoch, minibatch, total_minibatch,
                        train_error, test_error, self.best_cost, self.opt_parameters_message(opt_parameters)))

            else:
                if train_error < self.best_cost:
                    self.best_cost = train_error
                    self.best_rnn_fit = self.rnn.theta.get_value()

                logging.info('epoch {0}, minibatch {1}/{2}, tr loss {3}, {4}'.format(
                        epoch, minibatch, total_minibatch,
                        train_error, self.opt_parameters_message(opt_parameters)))

        return epoch_display

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


    def fit(self, x_train, y_train, x_test=None, y_test=None):
        """Fits self.rnn using self.optimizer and the provided options in __init__.
        :param x_train: training matrix with dimensions: 
                                        sequence length x #training samples x input dimensionality.
        :param y_train: training output matrix.
                            Dimensions: sequence length x #training samples x input dimensionality.
        :param x_test: test matrix with dimensions: 
                                        sequence length x #training samples x input dimensionality.
        :param y_test: test output matrix.
                            Dimensions: sequence length x #training samples x input dimensionality.
        """
        #Check for the availability of test set
        if x_test is not None:
            assert(y_test is not None)
            test_available = True
            test_set_x, test_set_y = self.shared_dataset((x_test, y_test))
        else:
            test_available = False

        #Loading train set as shared variable
        train_set_x, train_set_y = self.shared_dataset((x_train, y_train))
        self.best_cost = np.inf
        # compute number of minibatches for training
        # note that cases are the second dimension, not the first
        n_train = train_set_x.get_value(borrow=True).shape[1]
        n_train_batches = int(np.ceil(1.0 * n_train / self.batch_size))
        if test_available:
            n_test = test_set_x.get_value(borrow=True).shape[1]
            n_test_batches = int(np.ceil(1.0 * n_test / self.batch_size))

        cost = self.loss(self.y, self.rnn.y_t, burned_in=self.burned_in) \
            + self.L1_reg * self.rnn.L1 \
            + self.L2_reg * self.rnn.L2_sqr


        self.compute_error = theano.function(inputs=[self.rnn.x, self.y], 
                        outputs=self.loss(self.y, self.rnn.y_t,
                        burned_in=self.burned_in),
                        mode=self.mode)

        train_error_evolution = []
        test_error_evolution = []

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`
        self.best_rnn_fit = self.rnn.theta.get_value()
        epoch = 0
        if not test_available:
            call_back = self.get_call_back_function(x_train, y_train)
            train_error_evolution = self.optimizer.fit(self.rnn.x, self.y, train_set_x, train_set_y,
                            self.batch_size, cost, self.rnn.theta, self.epochs, self.compute_error, call_back)

        else:
            call_back = self.get_call_back_function(x_train, y_train, x_test=x_test, y_test=y_test)
            train_error_evolution, test_error_evolution = self.optimizer.fit(self.x, self.y, train_set_x, train_set_y,
                            self.batch_size, cost, self.rnn.theta, self.epochs, self.compute_error, call_back,
                            test_set_x=test_set_x, test_set_y=test_set_y)
        

        self.rnn.theta.set_value(self.best_rnn_fit)
        return train_error_evolution, test_error_evolution