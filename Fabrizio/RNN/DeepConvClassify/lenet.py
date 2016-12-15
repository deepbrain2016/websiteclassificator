import numpy
import theano
import theano.tensor as T

from convnet import ConvPoolLayer
from fullyconnectedneuralnetwork import FullyConnectedLayer 
from softmax import Softmax
from loss_functions import LossFunctions

# Define the LeCunn Deep Neural Convolutional Network
class Lenet(object):
    # First Layer - Convolutional 
    conv1_kernels = 20											  # Number of filters of the first layer	
    conv1_filter_height = 5										  # Filters height for the first convolutional layer	
    conv1_filter_width = 5										  # Filters width for the first convolutional layer
    conv1_pad = 0												  # Zero-padding for the first convolutional layer
    conv1_stride = 1											  # Stride for the first convolutional layer

    # Second Layer - Pool 
    conv1_poolsize = 2											  # Down-sampling pooling ratio for the first convolutional layer

    # Third Layer 
    conv2_kernels = 50 
    conv2_filter_height = 5										  # Filters height for the second convolutional layer	
    conv2_filter_width = 5										  # Filters width for the second convolutional layer
    conv2_pad = 0												  # Zero-padding for the second convolutional layer
    conv2_stride = 1											  # Stride for the second convolutional layer

    # Fourth Layer
    conv2_poolsize = 2											  # Down-sampling pooling ratio for the second convolutional layer

	# Activation Functions
    conv_activation = T.tanh
    fc_activation = T.nnet.sigmoid

    def __init__(self, index, x, y, batch_size, input_size, learning_rate, datasets):
        rng = numpy.random.RandomState(23455)
		
        # spread datasets onto all the input training sets, validation sets and testing sets
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
		
        filter_size = [self.conv1_filter_height, self.conv1_filter_width, self.conv2_filter_height, self.conv2_filter_width]
        pool_size = [self.conv1_poolsize, self.conv2_poolsize]
        padding = [self.conv1_pad, self.conv2_pad]
        stride = [self.conv1_stride, self.conv2_stride]
        nkerns=[self.conv1_kernels, self.conv2_kernels]
		
        layer0_input = x.reshape((batch_size, 1, input_size[0], input_size[1]))    			# Depth = 1, because the input image has no channels. H-eight=32, width = 32 which is the size of the input image, there are no channels.  
																							# 4d tensor is suitable for 1 ConvPoolLayer
        layer0 = ConvPoolLayer(                                    							# 1st conv + pool layer
            rng,                       			                             				# initial weights are random
            input=layer0_input,                         		            				# reshaped image is the input of the first layer
            image_shape=(batch_size,1,input_size[0],input_size[1]), 		   				# filtering reduces to the image size (32-5+1, 32-5+1) = (28, 28), image shape is 1x32x32    
            filter_shape=(nkerns[0],1,filter_size[0],filter_size[1]),        				# we have 20 filters of shape 1 (depth) x 5 x 5 
																							# max-pooling reduces to (28/2, 28/2) = (14,14), filter shape is 20x1x5x5
            pad = 0, 
            stride = 1, 
            poollayer = True, 
            poolsize=(pool_size[0],pool_size[0]),                                          	# max pooling is made on (2,2) areas of image, basically it's a non-linear down-sampling. 
																							# Basically for each region the maximum value is taken, so the input image si resized to the half of it
            activation=self.conv_activation
		)      
    
	    # Computes the first convolutional layer size 
        layer1_size = [((input_size[0]-filter_size[0]-2*padding[0])/stride[0]+1)/pool_size[0], ((input_size[1]-filter_size[1]-2*padding[0])/stride[0]+1)/pool_size[0],nkerns[0]]
	
        #increasing nkerns it should increase the parallelism power of GPU (number of kernels involved)
        layer1 = ConvPoolLayer(                                     						# 2nd conv + pool layer
            rng,                                                   	 						# initial weights are random
            input=layer0.output,                                    						# 2nd layer's input is the output from the first layer
            image_shape=(batch_size,nkerns[0],layer1_size[0],layer1_size[1]),            	# filtering reduces to the image size (14-5+1, 14-5+1) = (10, 10), image shape is 500x20x14x14, H=((32-5-2*0)/1+1)/poolsize = 14
            filter_shape=(nkerns[1],nkerns[0],filter_size[2],filter_size[3]),              	# now we have 50 filters which have size 20 (depth) x 5 x 5.
            pad = 0, 
            stride = 1, 
            poolsize=(pool_size[1],pool_size[1]),  								    		# pooling reduces to (10/2, 10/2) = (5,5)
            activation=self.conv_activation
        )                                                      
      
	    # Computes the fully connected layer input size 
        n_fc_inputs = int(nkerns[1] * ((layer1_size[0]-filter_size[2]-2*padding[1])/stride[1]+1)/pool_size[1] * ((layer1_size[1]-filter_size[3]-2*padding[1])/stride[1]+1)/pool_size[1])
	
		# Makes the input flat for the fc layer 
        layer2_input = layer1.output.flatten(2)                     						# generates a matrix batch_size, nkerns[1] * 5 * 5) for the hiddenlayer of the mlp, which is 50x5x5
 
        layer2 = FullyConnectedLayer(                               						# fully connected hidden layer
            rng,
            input=layer2_input,
            n_in=n_fc_inputs, 			 						                            # so the mlp reduces rom 1250 (50x4x4) neurons to 500 neurons
            n_out=500,
            activation=self.fc_activation
        )
    
		# Softmax layer: 500 inputs and 2 outputs
        layer3 = Softmax(input=layer2.output, n_in = 500, n_out=2)      					# change the output from 10 to 1 for our model, softmax reduces from 500 to 1 output
																							# final neural network has only 2 output
    
		# Loss function class (cost) 
        lf = LossFunctions(layer3) 
        cost = lf.negative_log_likelihood(y)
		
        self.test_model = theano.function(
            [index],
            layer3.errors(y),
            givens = {
                x: test_set_x[index * batch_size: (index+1)*batch_size],       				# takes as x the test_set for x of 1 batch_size
                y: test_set_y[index * batch_size: (index+1)*batch_size]          			# takes as y the test_set for x of 1 batch_size
            }
        )
    
        self.validate_model = theano.function(
            [index],
            layer3.errors(y),
            givens = {
                x: valid_set_x[index * batch_size: (index+1)*batch_size],
                y: valid_set_y[index * batch_size: (index+1)*batch_size]
            }
        )
    
        self.predict_model = theano.function(
            [index],
            outputs = [layer3.y_pred, y],
            givens = {
                x: test_set_x[index * batch_size: (index+1)*batch_size],    	    		# takes as x the test_set for x of 1 batch_size
                y: test_set_y[index * batch_size: (index+1)*batch_size]         	 		# takes as y the test_set for x of 1 batch_size
            }
        )
	
        self.params = layer3.params + layer2.params + layer1.params + layer0.params         		# list of symbolic gradients for all the model parameters. Trivially, from literature: the gradient of a shared weight 
	   	   																			        # is simply the sum of gradients of the parameters being shared

        grads = T.grad(cost, self.params)                                                 		# compute the gradients from errors   

        updates = [                                                                         # this is the back-propagation, updates all the parameters (weights and biases) by back-propagating gradients
            (param_i, param_i - learning_rate * grad_i)                                 	# times a learning_rate factor
            for param_i, grad_i in zip(self.params, grads)
        ]

        self.train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
		
        self.output = layer3.y_pred
		
	# Wraps of theano functions
    def train_model(self, index):
        self.train_model(index) 	
		
    def validate_model(self, index):
        self.validate_model(index) 	

    def test_model(self, index):
        self.test_model(index) 	

    def predict_model(self, index):
        self.predict_model(index)	