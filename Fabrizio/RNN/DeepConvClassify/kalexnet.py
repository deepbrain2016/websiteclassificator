import pdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

#AlexNet with batch normalization in Keras 
#input image is 224x224

class AlexNet:

    @staticmethod
    def build(width, height, depth, classes, mul_factor, weightsPath=None):
        model = Sequential()
        model.add(Convolution2D(96, 11, 11, border_mode='same', input_shape=(depth, height, width)))
        #model.add(BatchNormalization(mode=0, axis=1, input_shape=(depth, height, width)))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(Convolution2D(256, 5, 5, border_mode='same'))
        #model.add(BatchNormalization(mode=0, axis=1))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(Convolution2D(384, 3, 3, border_mode='same'))
        #model.add(BatchNormalization(mode=0, axis=1))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Activation('relu'))
        model.add(Convolution2D(384, 3, 3, border_mode='same'))
        #model.add(BatchNormalization(mode=0, axis=1))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Activation('relu'))
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        #model.add(BatchNormalization(mode=0, axis=1))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Activation('relu'))
		
        model.add(Flatten())
        model.add(Dense(4096, init = 'glorot_normal'))
        model.add(Dropout(0.5))
        #model.add(BatchNormalization(mode=0, axis=1))
        model.add(Activation('relu'))

        model.add(Dense(4096, init = 'glorot_normal'))
        model.add(Dropout(0.5))
        #model.add(BatchNormalization(mode=0, axis=1))
        model.add(Activation('relu'))

        model.add(Dense(2, init='glorot_normal'))
        model.add(Activation('softmax'))
        
        model.summary()

        #if a weights path is supplied (indicating that the model was pre-trained), then load the weights
        if weightsPath is not None: 
            model.load_wights(weightsPath)
			
        return model