
import theano
import numpy as np


if __name__ == '__main__':
    theano.sandbox.cuda.dnn.dnn_available.avail=True

    img = theano.tensor.tensor4()

    out = theano.sandbox.cuda.dnn.dnn_pool(img, (2,2))

    f = theano.function([img],out)
    d = np.random.rand(10,10,10,10).astype('float32')

    print (f(d).shape)