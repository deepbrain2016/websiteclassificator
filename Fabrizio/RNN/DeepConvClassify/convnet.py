import numpy
import theano
from theano import shared
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
#from theano.sandbox.cuda import dnn

import pdb

import theano.tensor as T


class ConvPoolLayer(object):


    def __init__ (self, rng, input, filter_shape, image_shape, pad, stride, poollayer=True, poolsize = (2,2), activation=T.tanh):

        self.input = input

        assert image_shape[1] == filter_shape[1]                            														# layer's depth and filter's must coincide           
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])                               														# "num input feature maps * filter height * filter width" inputs to the next layer

        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //       
                   numpy.prod(poolsize))                                    														# "num output feature maps * filter height * filter width" / pooling size        

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))                       														# initialize randomly weights to the conv layer
        self.W = theano.shared(                                             														# wrap weights in theano shared
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)  # bias is a 1d tensor 
        self.b = theano.shared(value=b_values, borrow=True)                     # wrap it in shared variables

        conv_out = conv2d(input=input, filters=self.W, filter_shape=filter_shape, input_shape=image_shape, subsample=(stride, stride))	    	# convolve input feature maps with filters
        
        #conv_out = dnn.dnn_conv(img=input, kerns=self.W, subsample=(stride, stride), border_mode=pad, algo='time_once')								# use of cuDNN library from NVIDIA CUDA

        if (poollayer==True):
            out = pool.pool_2d(input=conv_out, ds=poolsize, ignore_border=True) 													# pool each feature map individually, using maxpooling, pooling is made on 2 trailing dimensions of the input tensor, basically the last two 
        else: 
            out = conv_out																											# straight output
            poolsize=(1,1)																											# if no poollayer exists then the layer size stays the original	
			
        self.output = activation(out + self.b.dimshuffle('x', 0, 'x', 'x'))  														# apply activation, reshape b to add it to pooled_out
																																	# biases are converted in 4d tensor by dimshuffle, the new shaper of
																																	# biases is 1xAx1x1, basically its 4-dimensional row vector
																																	# this is neede to sum b to pooled_out which is a 4d tensor
		
        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
		
        self.size = [((image_shape[2]-filter_shape[2]-2*pad)/stride+1)/poolsize[0], ((image_shape[3]-filter_shape[3]-2*pad)/stride+1)/poolsize[0], filter_shape[0]]

        self.nneurons = self.size[0] * self.size[1] * self.size[2]

        self.nparams = filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]