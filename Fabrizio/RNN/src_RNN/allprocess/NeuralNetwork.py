'''
Created on 03/ott/2016

@author: fabrizio
'''

from Performance import *



import numpy as np
from  DataSet import *  
np.random.seed(1337) # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from Parameters import *


class NeuralNetwork(object):
    '''
    classdocs
    '''


    def __init__(self,mds):
        '''
        Constructor
        '''
        
        mydataset=mds
        print('Build model...')
        print ('setting local variable...')
        
        params=Parameters()
        self.type=params.TypeNN
        self.top_words=mydataset.max_features
        self.embedding_vecor_length=params.embedding_vecor_length
        self.nb_epoch=params.epoch
        self.N_LSTM=params.N_LSTM
        self.batch_size=params.batch_size
        self.ValidationSplit=params.ValidationSplit
        self.max_review_length=mydataset.max_site_length
        print ('load X,Y train test')
        (self.X_train, self.Y_train), (self.X_test, self.Y_test)=mydataset.dataSet
        print ("load model")
        self.model()
     
    def learning(self):
        FormatModelSaved="./lstm_saved.{epoch:02d}-{val_loss:.2f}.hdf5"

        def f1(y_true, y_pred):
        
            SUM=((y_pred-y_true)**2).sum(keepdims=True)   
        
            PROD=((y_pred*y_true)).sum(keepdims=True)   
        
            loss=-1/(1+(SUM/(2*(PROD+0.00001))))
    
            return loss
        
        #self.model.compile(loss='binary_crossentropy',
        #              optimizer='adam',
        #              metrics=['accuracy'])

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        
        print('Train...')
        checkPoint=ModelCheckpoint(FormatModelSaved, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')

        self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                validation_split=self.ValidationSplit, callbacks=[checkPoint])
                #  validation_data=(self.X_test, self.Y_test))

        
        return None
         
    def testing(self):
        score, acc = self.model.evaluate(self.X_test, self.Y_test,
                            batch_size=self.batch_size)
        
        
        prediction= (self.model.predict_classes(self.X_test, verbose=2))



        
#         print('Test score:', score)
#         print('Test accuracy:', acc)  
        P=Performance(score,acc,prediction,self.Y_test)
        return P
       
    def model (self):

        if (self.type=='LSTM'):
            self.model = Sequential()
            self.model.add(Embedding(self.top_words, self.embedding_vecor_length, input_length=self.max_review_length))
            self.model.add(LSTM(self.N_LSTM)) # try using a GRU instead, for fun
            self.model.add(Dense(1,activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        if (self.type=="CNNLSTM"):
            self.model = Sequential()
            self.model.add(Embedding(self.top_words, self.embedding_vecor_length, input_length=self.max_review_length))
            self.model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
            self.model.add(MaxPooling1D(pool_length=2))
            self.model.add(LSTM(self.N_LSTM))
            self.model.add(Dense(1, activation='sigmoid'))


        if (self.type=="V2WCNNLSTM"):
            self.model = Sequential()
            #self.model.add(Embedding(self.top_words, self.embedding_vecor_length, input_length=self.max_review_length))
            self.model.add(Convolution1D(input_shape=(self.max_review_length, self.embedding_vecor_length),nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
            self.model.add(MaxPooling1D(pool_length=2))
            self.model.add(LSTM(self.N_LSTM))
            self.model.add(Dense(1, activation='sigmoid'))



            
