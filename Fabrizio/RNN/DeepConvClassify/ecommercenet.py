import numpy
import theano
import theano.tensor as T
import scipy.misc
import pdb

from convnet import ConvPoolLayer
from fullyconnectedneuralnetwork import FullyConnectedLayer 
from softmax import Softmax
from loss_functions import LossFunctions
from training_algorithms import TrainingAlgs

# Define the LeCunn Deep Neural Convolutional Network
class Ecommercenet(object):
    
	# Activation Functions
    conv_activation = T.tanh
    fc_activation = T.nnet.sigmoid
	
    def __init__(self, index, x, y, batch_size, input_size, learning_rate, datasets, mul_factor):
    
        rng = numpy.random.RandomState(23455)
		
        # spread datasets onto all the input training sets, validation sets and testing sets
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
		
        filter_size = [5, 5, 5, 5]
        pool_size = [2, 2]
        padding = [0, 0]
        stride = [1, 1]
        nkerns=[20, 50]
        #nkerns=[80, 160, 320, 640, 1280, 2560]
		
        layer0_input = x.reshape((batch_size, 1, input_size[0], input_size[1]))    			# Depth = 1, because the input image has no channels. H-eight=32, width = 32 which is the size of the input image, there are no channels.  
																							# 4d tensor is suitable for 1 ConvPoolLayer
        totalnneurons = 0
        totalnparams = 0
		
        print('\nConvolutional layers: ')
        # Convolutional layers
        # layer0: 28x28x20
        
        '''
        layer0 = ConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size,1,input_size[0],input_size[1]), filter_shape=(nkerns[0],1,filter_size[0],filter_size[1]), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += layer0.nneurons
        totalnparams += layer0.nparams
        print('Layer 0 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (layer0.size[0], layer0.size[1], layer0.size[2], layer0.nneurons, layer0.nparams))
		# layer1: 24x24x40
        layer1 = ConvPoolLayer(rng, input=layer0.output, image_shape=(batch_size,nkerns[0],layer0.size[0], layer0.size[1]), filter_shape=(nkerns[1],nkerns[0],filter_size[0],filter_size[1]), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += layer1.nneurons
        totalnparams += layer1.nparams
        print('Layer 1 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (layer1.size[0], layer1.size[1], layer1.size[2], layer1.nneurons, layer1.nparams))
		# layer2: 20x20x80
        layer2 = ConvPoolLayer(rng, input=layer1.output, image_shape=(batch_size,nkerns[1],layer1.size[0], layer1.size[1]), filter_shape=(nkerns[2],nkerns[1],filter_size[0],filter_size[1]), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += layer2.nneurons
        totalnparams += layer2.nparams
        print('Layer 2 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (layer2.size[0], layer2.size[1], layer2.size[2], layer2.nneurons, layer2.nparams))
		# layer3: 16x16x160
        layer3 = ConvPoolLayer(rng, input=layer2.output, image_shape=(batch_size,nkerns[2],layer2.size[0], layer2.size[1]), filter_shape=(nkerns[3],nkerns[2],filter_size[0],filter_size[1]), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += layer3.nneurons
        totalnparams += layer3.nparams
        print('Layer 3 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (layer3.size[0], layer3.size[1], layer3.size[2], layer3.nneurons, layer3.nparams))
		# layer4: 12x12x320
        layer4 = ConvPoolLayer(rng, input=layer3.output, image_shape=(batch_size,nkerns[3],layer3.size[0], layer3.size[1]), filter_shape=(nkerns[4],nkerns[3],filter_size[0],filter_size[1]), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += layer4.nneurons
        totalnparams += layer4.nparams
        print('Layer 4 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (layer4.size[0], layer4.size[1], layer4.size[2], layer4.nneurons, layer4.nparams))
		# layer5: 8x8x640
        layer5 = ConvPoolLayer(rng, input=layer4.output, image_shape=(batch_size,nkerns[4],layer4.size[0], layer4.size[1]), filter_shape=(nkerns[5],nkerns[4],filter_size[0],filter_size[1]), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += layer5.nneurons
        totalnparams += layer5.nparams
        print('Layer 5 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (layer5.size[0], layer5.size[1], layer5.size[2], layer5.nneurons, layer5.nparams))
        '''
		
		# Convolutional special layer 1 - with one layer f=0.522
        convlayer1 = ConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size,1,input_size[0],input_size[1]), filter_shape=(92,1,mul_factor*21,mul_factor*21), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += convlayer1.nneurons
        totalnparams += convlayer1.nparams
        print('Conv Layer 1 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer1.size[0], convlayer1.size[1], convlayer1.size[2], convlayer1.nneurons, convlayer1.nparams))

        '''
        # Convolutional special layer 1 - decomposition
        convlayer1_1 = ConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size,1,input_size[0],input_size[1]), filter_shape=(72,1,3,3), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += convlayer1_1.nneurons
        totalnparams += convlayer1_1.nparams
        print('Conv Layer 1_1 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer1_1.size[0], convlayer1_1.size[1], convlayer1_1.size[2], convlayer1_1.nneurons, convlayer1_1.nparams))
        
        convlayer1_2 = ConvPoolLayer(rng, input=convlayer1_1.output, image_shape=(batch_size,72,convlayer1_1.size[0],convlayer1_1.size[0]), filter_shape=(82,72,3,3), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += convlayer1_2.nneurons
        totalnparams += convlayer1_2.nparams
        print('Conv Layer 1_2 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer1_2.size[0], convlayer1_2.size[1], convlayer1_2.size[2], convlayer1_2.nneurons, convlayer1_2.nparams))
		
        convlayer1_3 = ConvPoolLayer(rng, input=convlayer1_2.output, image_shape=(batch_size,82,convlayer1_2.size[0],convlayer1_2.size[0]), filter_shape=(92,82,3,3), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += convlayer1_3.nneurons
        totalnparams += convlayer1_3.nparams
        print('Conv Layer 1_3 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer1_3.size[0], convlayer1_3.size[1], convlayer1_3.size[2], convlayer1_3.nneurons, convlayer1_3.nparams))
		
        convlayer1_4 = ConvPoolLayer(rng, input=convlayer1_3.output, image_shape=(batch_size,92,convlayer1_3.size[0],convlayer1_3.size[0]), filter_shape=(102,92,3,3), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += convlayer1_4.nneurons
        totalnparams += convlayer1_4.nparams
        print('Conv Layer 1_4 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer1_4.size[0], convlayer1_4.size[1], convlayer1_4.size[2], convlayer1_4.nneurons, convlayer1_4.nparams))
        		
        convlayer1_5 = ConvPoolLayer(rng, input=convlayer1_4.output, image_shape=(batch_size,102,convlayer1_4.size[0],convlayer1_4.size[0]), filter_shape=(112,102,11,11), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += convlayer1_5.nneurons
        totalnparams += convlayer1_5.nparams
        print('Conv Layer 1_5 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer1_5.size[0], convlayer1_5.size[1], convlayer1_5.size[2], convlayer1_5.nneurons, convlayer1_5.nparams))

        '''
        '''
		convlayer1_6 = ConvPoolLayer(rng, input=convlayer1_5.output, image_shape=(batch_size,72,convlayer1_5.size[0],convlayer1_5.size[0]), filter_shape=(72,72,3,3), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += convlayer1_6.nneurons
        totalnparams += convlayer1_6.nparams
        print('Conv Layer 1_6 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer1_6.size[0], convlayer1_6.size[1], convlayer1_6.size[2], convlayer1_6.nneurons, convlayer1_6.nparams))

        convlayer1_7 = ConvPoolLayer(rng, input=convlayer1_6.output, image_shape=(batch_size,72,convlayer1_6.size[0],convlayer1_6.size[0]), filter_shape=(72,72,3,3), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += convlayer1_7.nneurons
        totalnparams += convlayer1_7.nparams
        print('Conv Layer 1_7 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer1_7.size[0], convlayer1_7.size[1], convlayer1_7.size[2], convlayer1_7.nneurons, convlayer1_7.nparams))

        convlayer1_8 = ConvPoolLayer(rng, input=convlayer1_7.output, image_shape=(batch_size,72,convlayer1_7.size[0],convlayer1_7.size[0]), filter_shape=(72,72,3,3), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += convlayer1_8.nneurons
        totalnparams += convlayer1_8.nparams
        print('Conv Layer 1_8 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer1_8.size[0], convlayer1_8.size[1], convlayer1_8.size[2], convlayer1_8.nneurons, convlayer1_8.nparams))

        convlayer1_9 = ConvPoolLayer(rng, input=convlayer1_8.output, image_shape=(batch_size,72,convlayer1_8.size[0],convlayer1_8.size[0]), filter_shape=(72,72,3,3), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += convlayer1_9.nneurons
        totalnparams += convlayer1_9.nparams
        print('Conv Layer 1_9 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer1_9.size[0], convlayer1_9.size[1], convlayer1_9.size[2], convlayer1_9.nneurons, convlayer1_9.nparams))
        '''
        
		# Convolutional special layer 2 - Good layer f=0.527
        convlayer2 = ConvPoolLayer(rng, input=convlayer1.output, image_shape=(batch_size,92,convlayer1.size[0],convlayer1.size[1]), filter_shape=(72,92,mul_factor*3,mul_factor*3), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += convlayer2.nneurons
        totalnparams += convlayer2.nparams
        print('Conv Layer 2 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer2.size[0], convlayer2.size[1], convlayer2.size[2], convlayer2.nneurons, convlayer2.nparams))

       
        #convlayer3 = ConvPoolLayer(rng, input=convlayer2.output, image_shape=(batch_size,72,convlayer2.size[0],convlayer2.size[1]), filter_shape=(72,72,1,1), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        #totalnneurons += convlayer3.nneurons
        #totalnparams += convlayer3.nparams
        #print('Conv Layer 3 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer3.size[0], convlayer3.size[1], convlayer3.size[2], convlayer3.nneurons, convlayer3.nparams))

        #convlayer4 = ConvPoolLayer(rng, input=convlayer3.output, image_shape=(batch_size,72,convlayer3.size[0],convlayer3.size[1]), filter_shape=(72,72,1,1), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        #totalnneurons += convlayer4.nneurons
        #totalnparams += convlayer4.nparams
        #print('Conv Layer 4 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer4.size[0], convlayer4.size[1], convlayer4.size[2], convlayer4.nneurons, convlayer4.nparams))

        ''' Experimenting still problems 
		convlayer2 = ConvPoolLayer(rng, input=convlayer1.output, image_shape=(batch_size,72,convlayer1.size[0],convlayer1.size[1]), filter_shape=(144,72,8,8), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        totalnneurons += convlayer2.nneurons
        totalnparams += convlayer2.nparams
        print('Conv Layer 2 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer2.size[0], convlayer2.size[1], convlayer2.size[2], convlayer2.nneurons, convlayer2.nparams))
        
        #convlayer3 = ConvPoolLayer(rng, input=convlayer2.output, image_shape=(batch_size,144,convlayer2.size[0],convlayer2.size[1]), filter_shape=(288,144,3,3), pad = 0, stride = 1, poollayer=False, poolsize=(pool_size[0],pool_size[0]), activation=self.conv_activation)  
        #totalnneurons += convlayer2.nneurons
        #totalnparams += convlayer2.nparams
        #print('Conv Layer 3 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (convlayer3.size[0], convlayer3.size[1], convlayer3.size[2], convlayer3.nneurons, convlayer3.nparams))
        '''
		
		# Set the last conv layer as input for the fully connected layer
        #fc1_inputlayer = layer0	
        fc1_inputlayer = convlayer2		
        #fc1_inputlayer = layer5	
        #fc1_inputlayer = layer0_input
		
		
	    # Computes the fully connected layer input size 
        n_fc1_inputs = int(fc1_inputlayer.size[0] * fc1_inputlayer.size[1] * fc1_inputlayer.size[2])
	
        # Makes the input flat for the fc layer 
        fc1_input = fc1_inputlayer.output.flatten(2) 	                   						# generates a matrix batch_size, nkerns[1] * 5 * 5) for the hiddenlayer of the mlp, which is 50x5x5
        
        #n_fc1_inputs = input_size[0] * input_size[1]
	
        #fc1_input = layer0_input.flatten(2) 

        fc1_layer = FullyConnectedLayer(rng, input=fc1_input, n_in=n_fc1_inputs, n_out=500, activation=self.fc_activation)   
        totalnneurons += fc1_layer.nneurons
        totalnparams += fc1_layer.nparams
        print('Fully Connected Layer 1 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (fc1_layer.size[0], fc1_layer.size[1], fc1_layer.size[2], fc1_layer.nneurons, fc1_layer.nparams))
        #fc2_layer = FullyConnectedLayer(rng, input=fc1_layer.output, n_in=fc1_layer.size[2], n_out=500, activation=self.fc_activation)
        #totalnneurons += fc2_layer.nneurons
        #totalnparams += fc2_layer.nparams
        #print('Fully Connected Layer 2 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (fc2_layer.size[0], fc2_layer.size[1], fc2_layer.size[2], fc2_layer.nneurons, fc2_layer.nparams))
        #fc3_layer = FullyConnectedLayer(rng, input=fc2_layer.output, n_in=fc2_layer.size[2], n_out=1024, activation=self.fc_activation)
        #totalnneurons += fc3_layer.nneurons
        #totalnparams += fc3_layer.nparams
        #print('Fully Connected Layer 2 - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (fc3_layer.size[0], fc3_layer.size[1], fc3_layer.size[2], fc3_layer.nneurons, fc3_layer.nparams))
    
		# Set the last fully connected layer as input for the final layer (softmaex, ecc.) 
        secondtolast_layer = fc1_layer

		# Softmax layer: 500 inputs and 2 outputs
        ninput_final_layer = 500
        noutput_final_layer = 2
        final_layer = Softmax(input=secondtolast_layer.output, n_in = ninput_final_layer, n_out=noutput_final_layer)   				# change the output from 10 to 1 for our model, softmax reduces from 500 to 1 output
																																	# final neural network has only 2 output
        totalnneurons += noutput_final_layer
        totalnparams += ninput_final_layer * noutput_final_layer
        print('Final Layer - Size: %ix%ix%i - Number of Neurons: %i - Number of Weights: %i' % (1, 1, ninput_final_layer, noutput_final_layer, ninput_final_layer * noutput_final_layer))

        print('\n Total number of Parameters of the Deep Neural Network: ', int(totalnparams)) 
        print('\n Total number of Neurons of the Deep Neural Network: ', int(totalnneurons)) 
		
		# Loss function class (cost) 
        lf = LossFunctions(final_layer) 
        cost = lf.negative_log_likelihood(y)
		
        self.test_model = theano.function(
            [index],
            final_layer.errors(y),
            givens = {
                x: test_set_x[index * batch_size: (index+1)*batch_size],       				# takes as x the test_set for x of 1 batch_size
                y: test_set_y[index * batch_size: (index+1)*batch_size]          			# takes as y the test_set for x of 1 batch_size
            }
        )
    
        self.validate_model = theano.function(
            [index],
            final_layer.errors(y),
            givens = {
                x: valid_set_x[index * batch_size: (index+1)*batch_size],
                y: valid_set_y[index * batch_size: (index+1)*batch_size]
            }
        )
    
        self.predict_model = theano.function(
            [index],
            outputs = [final_layer.y_pred, y],
            givens = {
                x: test_set_x[index * batch_size: (index+1)*batch_size],    	    		# takes as x the test_set for x of 1 batch_size
                y: test_set_y[index * batch_size: (index+1)*batch_size]         	 		# takes as y the test_set for x of 1 batch_size
            }
        )
	
        #self.params = final_layer.params + fc1_layer.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params        		# list of symbolic gradients for all the model parameters. Trivially, from literature: the gradient of a shared weight 
        self.params = final_layer.params + fc1_layer.params + convlayer2.params + convlayer1.params        		# list of symbolic gradients for all the model parameters. Trivially, from literature: the gradient of a shared weight 
        #self.params = final_layer.params + fc1_layer.params + convlayer2.params + convlayer1_5.params + convlayer1_4.params + convlayer1_3.params + convlayer1_2.params + convlayer1_1.params        		# list of symbolic gradients for all the model parameters. Trivially, from literature: the gradient of a shared weight 
        #self.params = final_layer.params + fc1_layer.params + convlayer2.params + convlayer1.params        		# list of symbolic gradients for all the model parameters. Trivially, from literature: the gradient of a shared weight 
        #self.params = final_layer.params + fc2_layer.params + fc1_layer.params + convlayer1.params      		# list of symbolic gradients for all the model parameters. Trivially, from literature: the gradient of a shared weight 
	   	   																			        # is simply the sum of gradients of the parameters being shared
        #pdb.set_trace()
		

        self.train_model = theano.function(
            [index],
            cost,
            updates=TrainingAlgs.sgd(loss=cost, all_params=self.params, learning_rate=learning_rate),	# Training algorithm SGD
            #updates=TrainingAlgs.adam(loss=cost, all_params=self.params, learning_rate=learning_rate),	# Training algorithm ADAM
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
		
        self.output = final_layer.y_pred
		
	# Wraps of theano functions
    def train_model(self, index):
        self.train_model(index) 	
		
    def validate_model(self, index):
        self.validate_model(index) 	

    def test_model(self, index):
        self.test_model(index) 	

    def predict_model(self, index):
        self.predict_model(index)	