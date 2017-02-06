# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
class MyModels:
	@staticmethod
	def getNetTypes():
		print "richiesta tipi di reti"
		list_net=['mpl','mpl3','lenet','lenet500','lenet500_huge','lenetREC','lenet3','lenetVisualScr']
		print list_net
		return list_net
	@staticmethod
	def getNetInDim(NNetType):
		print "richiesta del numero di dimensioni dell'input..."
		if (NNetType=='mlp'):
			return None
		if (NNetType=='mlp3'):
			return None
		if (NNetType=='lenet500'):
			return 32
		if (NNetType=='lenet500_huge'):
			return 32
		if (NNetType=='lenet'):
			return 32
		if (NNetType=='lenet3'):
			return 32
		if (NNetType=='lenetREC'):
			return 24
		if (NNetType=='lenetVisualScr'):
			return 64
		print "non ci sono modelli selezionati"		
	@staticmethod
	def getInputDIM(NNetType):
		print "richiesta del numero di dimensioni dell'input..."
		if (NNetType=='mlp'):
			return 1
		if (NNetType=='mlp3'):
			return 1
		if (NNetType=='lenet500'):
			return 2
		if (NNetType=='lenet500_huge'):
			return 2
		if (NNetType=='lenet'):
			return 2
		if (NNetType=='lenet3'):
			return 2
		if (NNetType=='lenetREC'):
			return 2
		if (NNetType=='lenetVisualScr'):
			return 2
		print "non ci sono modelli selezionati"				
	@staticmethod
	def getNClasses(NNetType):
		print "richiesta del numero di classi di output del modello..."
		if (NNetType=='mlp'):
			return 1
		if (NNetType=='mlp3'):
			return 1
		if (NNetType=='lenet'):
			return 2
		if (NNetType=='lenet500'):
			return 2
		if (NNetType=='lenet500_huge'):
			return 2
		if (NNetType=='lenet3'):
			return 2
		if (NNetType=='lenetREC'):
			return 2
		if (NNetType=='lenetVisualScr'):
			return 2
		print "non ci sono modelli selezionati"				
	@staticmethod
	def build(NNetType):
		if (NNetType=='mlp'):
			print "Seleziona il modello mpl 1000 12 8 1"
			model = Sequential()
			model.add(Dense(12, input_dim=1000, init='uniform', activation='relu'))
			model.add(Dense(8, init='uniform', activation='relu'))
			model.add(Dense(1, init='uniform', activation='sigmoid'))
			return model
		if (NNetType=='mlp3'):
			print "Seleziona il modello mpl3 3000 12 8 1"
			model = Sequential()
			model.add(Dense(12, input_dim=1000*3, init='uniform', activation='relu'))
			model.add(Dense(8, init='uniform', activation='relu'))
			model.add(Dense(1, init='uniform', activation='sigmoid'))
			return model
		if (NNetType=='lenet500'):
			print "Seleziona il modello lenet 32 20_5 50_5 50"
			model = Sequential()
			# first set of CONV => RELU => POOL
			model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(1,32, 32)))
			model.add(Activation("tanh"))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			# selcond set of CONV => RELU => POOL
			model.add(Convolution2D(50, 5, 5, border_mode="same"))
			model.add(Activation("tanh"))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			# third set MPL
			model.add(Flatten())
			model.add(Dense(50))
			model.add(Activation("tanh"))
			# softmax classifier
			model.add(Dense(2))
			model.add(Activation("softmax"))
			return model
		if (NNetType=='lenet500_huge'):
			print "Seleziona il modello lenet 32 20_15 50_15 500"
			model = Sequential()
			# first set of CONV => RELU => POOL
			model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(1,32, 32)))
			model.add(Activation("tanh"))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			# selcond set of CONV => RELU => POOL
			model.add(Convolution2D(50, 5, 5, border_mode="same"))
			model.add(Activation("tanh"))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			# third set MPL
			model.add(Flatten())
			model.add(Dense(500))
			model.add(Activation("tanh"))
			# softmax classifier
			model.add(Dense(2))
			model.add(Activation("softmax"))
			return model
		if (NNetType=='lenet'):
			print "Seleziona il modello lenet 32 20_5 50_2 50"
			model = Sequential()
			# first set of CONV => RELU => POOL
			model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(1,32, 32)))
			model.add(Activation("tanh"))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			# selcond set of CONV => RELU => POOL
			model.add(Convolution2D(50, 5, 5, border_mode="same"))
			model.add(Activation("tanh"))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			# third set MPL
			model.add(Flatten())
			model.add(Dense(50))
			model.add(Activation("tanh"))
			# softmax classifier
			model.add(Dense(2))
			model.add(Activation("softmax"))
			return model
		if (NNetType=='lenetREC'):
			print "Seleziona il modello lenet 24 20_5 50_2 50"
			model = Sequential()
			# first set of CONV => RELU => POOL
			model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(1,24, 24)))
			model.add(Activation("tanh"))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			# selcond set of CONV => RELU => POOL
			model.add(Convolution2D(50, 5, 5, border_mode="same"))
			model.add(Activation("tanh"))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			# third set MPL
			model.add(Flatten())
			model.add(Dense(50))
			model.add(Activation("tanh"))
			# softmax classifier
			model.add(Dense(2))
			model.add(Activation("softmax"))
			return model
		if (NNetType=='lenet3'):
			print "Seleziona il modello lenet 3*32 20_5 50_5 50"
			model = Sequential()
			# first set of CONV => RELU => POOL
			model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(3,32, 32)))
			model.add(Activation("tanh"))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			# selcond set of CONV => RELU => POOL
			model.add(Convolution2D(50, 5, 5, border_mode="same"))
			model.add(Activation("tanh"))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			# third set MPL
			model.add(Flatten())
			model.add(Dense(50))
			model.add(Activation("tanh"))
			# softmax classifier
			model.add(Dense(2))
			model.add(Activation("softmax"))
			return model
		if (NNetType=='lenetVisualScr'):
			print "Seleziona il modello lenet 3*64 20_5 50_5 100"
			model = Sequential()
			# first set of CONV => RELU => POOL
			model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(3,64,64)))
			model.add(Activation("tanh"))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			# selcond set of CONV => RELU => POOL
			model.add(Convolution2D(50, 5, 5, border_mode="same"))
			model.add(Activation("tanh"))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			# third set MPL
			model.add(Flatten())
			model.add(Dense(100))
			model.add(Activation("tanh"))
			# softmax classifier
			model.add(Dense(2))
			model.add(Activation("softmax"))
			return model
		print "non ci sono modelli selezionati"
