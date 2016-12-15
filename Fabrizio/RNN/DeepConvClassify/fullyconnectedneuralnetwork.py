import numpy
import theano
import theano.tensor as T

# Define the last fully connected hidden layer
class FullyConnectedLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):

        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),                       # suggested configuration of weights
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4                                                   # in case activation function is sigmoid it's suggested to mul weights * 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b                              # Network spread
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
		
        self.size = [1,1,n_out]
		
        self.nneurons = n_out
		
        self.nparams = n_in * n_out

