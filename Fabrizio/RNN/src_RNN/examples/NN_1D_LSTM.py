'''
Created on 20/ott/2016

@author: fabrizio
'''

# LSTM and CNN for sequence classification in the IMDB dataset
import numpy,sys
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)



outfile=open("./outfile",'w')
outfile.write(str(X_train[10])+"len"+str(len(X_train[10]))+'\n')
outfile.write(str(X_train[100])+"len"+str(len(X_train[100]))+'\n')
outfile.write(str(X_train[1000])+"len"+str(len(X_train[1000]))+'\n')
# truncate and pad input sequences
max_review_length = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
outfilepad=open("./outfile_pad",'w')
outfilepad.write(str(X_train[10])+"len"+str(len(X_train[10]))+'\n')
outfilepad.write(str(X_train[100])+"len"+str(len(X_train[100]))+'\n')
outfilepad.write(str(X_train[1000])+"len"+str(len(X_train[1000]))+'\n')


sys.exit(0)
# create the model

embedding_vecor_length = 32

model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
