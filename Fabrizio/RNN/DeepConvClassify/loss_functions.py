import theano
import theano.tensor as T

# Define all the type of loss functions 
class LossFunctions(object):

    def __init__(self, input):
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.input.p_y_given_x)[T.arange(y.shape[0]), y])                # negative log-likelihood of the prediction
