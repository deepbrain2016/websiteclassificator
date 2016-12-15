import numpy
import theano
import theano.tensor as T

# Define the last layer which is softmax
class Softmax(object):

    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)                #W is the set of hyperplanes

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)                                #take the max probability

        self.params = [self.W, self.b]

        self.input = input

    def errors(self, y):

        if y.ndim != self.y_pred.ndim:                                                   # check if y has same dimension of y_pred
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startswith('int'):                                                   # check if y is of the correct datatype
            return T.mean(T.neq(self.y_pred, y))                                        # the T.neq operator returns a vector of 0s and 1s, where 1
                                                                                        # represents a mistake in prediction
            #return T.sum(T.neq(self.y_pred, y))                                        # the T.neq operator returns a vector of 0s and 1s, where 1

        else:
            raise NotImplementedError()
