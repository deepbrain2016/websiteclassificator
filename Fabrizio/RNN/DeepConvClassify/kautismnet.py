from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
import pdb

class AutismNet:

    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
	    # Activation Functions
        conv_activation = "tanh"
        fc_activation = "sigmoid"

        # initialize the model
        model = Sequential()
        # first set of CONV => RELU => POOL
        model.add(Convolution2D(92, 21, 21, border_mode="same", input_shape=(depth, height, width)))
        model.add(Activation(conv_activation))
		
        # second set of CONV => RELU => POOL
        model.add(Convolution2D(72, 3, 3, border_mode="same"))
        model.add(Activation(conv_activation))
		
		#set of FC => RELU Layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(fc_activation))

        model.summary()

		#softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
		
        #if a weights path is supplied (indicating that the model was pre-trained), then load the weights
        if weightsPath is not None: 
            model.load_wights(weightsPath)
			
        return model