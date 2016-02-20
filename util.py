import theano
import theano.tensor as T
import io
import json

def get_activation_function(activation):

    activations = ['tanh', 'sigmoid', 'relu', 'linear', 'cappedrelu']

    if activation == activations[0]:
        return T.tanh
    elif activation == activations[1]:
        return T.nnet.sigmoid
    elif activation == activations[2]:
        return lambda x: x * (x > 0)
    elif activation == activations[3]:
        return lambda x: x
    elif activation == activations[4]:
        return lambda x: T.minimum(x * (x > 0), 6)
    else:
        raise NotImplementedError, \
        "Activation function not implemented. Choose one out of: {0}".format(activations)



def parse_activations(activation_list):
    """From list of activation names for each layer return a list with the activation functions"""

    activation = [None]*len(activation_list)

    for i, act in enumerate(activation_list):
        activation[i] = get_activation_function(act)

    return activation


def load_json(filename):
    with io.open('{0}'.format(filename),
                 encoding='utf-8') as f:
        return json.load(f)


def save_json(filename, data):
    with io.open('{0}'.format(filename),
                 'w', encoding='utf-8') as f:
        f.write(unicode(json.dumps(data, ensure_ascii=False)))


def mse(y, y_pred, burned_in=0):
    return T.mean((y_pred[burned_in:] - y[burned_in:]) ** 2)

def proportions(y, y_pred, burned_in=0):
    return T.mean((y_pred[burned_in:-1]*1./y_pred[(burned_in+1):] - y[burned_in:-1]*1./y[(burned_in+1):]) ** 2)

def define_loss(loss_name):

    implemented_loss = ['mse', 'proportions']
    if loss_name == implemented_loss[0]:
        return mse
    elif loss_name == implemented_loss[1]:
        return proportions
    else:
        raise NotImplementedError("Not understandable loss function: {0!r} Supported losses: {1}", implemented_loss)
