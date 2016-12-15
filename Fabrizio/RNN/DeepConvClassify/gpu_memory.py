import numpy
import theano
import theano.tensor as T

def shared_dataset(data_xy, borrow=True):                                  # shared variables are useful to avoid delays for minibatch copies to the GPU
    data_x, data_y = data_xy
        
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

    return shared_x, T.cast(shared_y, 'int32')
