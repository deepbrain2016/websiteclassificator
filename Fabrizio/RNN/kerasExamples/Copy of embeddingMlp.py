'''
Created on 03/gen/2017

@author: fabrizio
'''



'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

#from __future__ import print_function
import theano.tensor as T
import theano
import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, load_model
import sys


def f1(y_true, y_pred):
    a1=np.asarray([1,0])
    a2=np.asarray([0,1])
    #(1,0)(0.34,0.7676)
    
    
    v1=T.dot(y_true,a1)
    v2=T.dot(y_true,a2)

    u1=T.dot(y_pred,a1)
    u2=T.dot(y_pred,a2)

    sv=(T.tanh((v2/v1-1)*1)+1)/2#*1000)
        
    su=(T.tanh((u2/u1-1)*1)+1)/2#*1000)
    
    SUM=((sv-su)**2).sum(keepdims=True)  
    
    
    #return T.mean(T.square(y_pred - y_true), axis=-1)
    

    PROD=((sv*su)).sum(keepdims=True)   
    
    loss=-1/(1+(SUM/(2*(PROD+0.00001))))
    
    

    return loss


def confusion_matrix(TestID_Prediction,Y_test):
        NClasses=2
        TP=0
        TN=0
        FP=0
        FN=0
        i=0
        elencoTN=[]
        elencoTP=[]
        elencoFN=[]
        elencoFP=[]        
        
    
        for y_p in TestID_Prediction:

            if NClasses==1:

                if y_p[0]==1 and Y_test[i]==1: 
                    TP=TP+1
                    #elencoTP.append(testID)
                if y_p[0]==0 and Y_test[i]==0: 
                    TN=TN+1
                    #elencoTN.append(testID) 
                if y_p[0]==0 and Y_test[i]==1: 
                    FN=FN+1 
                    #elencoFN.append(testID)
                if y_p[0]==1 and Y_test[i]==0: 
                    FP=FP+1 
                    #elencoFP.append(testID)
                
            if NClasses==2:      
                #print y_p[0]  , y_p[1]
                
                if y_p[0]>0.5:
                    y_p_class=0
                else:
                    y_p_class=1
                    
                    
                
                if y_p_class==1 and np.dot(Y_test[i],[0,1])==1: 
                    TP=TP+1
                    #elencoTP.append(testID)
                if y_p_class==0 and np.dot(Y_test[i],[0,1])==0: 
                    TN=TN+1 
                    #elencoTN.append(testID)
                if y_p_class==0 and np.dot(Y_test[i],[0,1])==1: 
                    FP=FP+1 
                    #elencoFP.append(testID)
                if y_p_class==1 and np.dot(Y_test[i],[0,1])==0: 
                    FN=FN+1 
                    #elencoFN.append(testID)
                
            i=i+1
        elencoTN=elencoTN
        elencoTP=elencoTP
        elencoFN=elencoFN
        elencoFP=elencoFP
    
        print "TP",TP
        print "TN",TN
        print "FP",FP
        print "FN",FN
    
        F1=2*float(TP)/(2*float(TP)+float(FN)+FP)
        ACC=(float(TP)+TN)/(TN+TP+FN+FP)
        try:
            PREC=float(TP)/(TP+FP)
        except:
            PREC=99999
        try:
            REC=float(TP)/(TP+FN)
        except:
            PREC=99999
            
    
        
        
    
        print "F1: "+str(F1)
        print "PREC: "+str(PREC)
        print "REC: "+str(REC)
        print "ACC: "+str(ACC)
        return TP,TN,FP,FN,F1,PREC,REC,ACC
    


mode='l'

mode='s'


BASE_DIR = './'
#GLOVE_DIR = BASE_DIR + '/glove.6B/'
#TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'

W2V_DIR = BASE_DIR + '/word2vecEC/'

TEXT_DATA_DIR = BASE_DIR + '/test/'
TEXT_DATA_DIR = BASE_DIR + '/EcommerceTest/'
TEXT_DATA_DIR = BASE_DIR + '/Ecommerce/'



MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 18000
EMBEDDING_DIM = 100


VALIDATION_SPLIT = 0.1

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')


##### CARICA IN EMBEDDINGS_INDEX 
embeddings_index = {}
f = open(os.path.join(W2V_DIR, 'word2vecDict.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

    
def get_data(TEXT_DATA_DIR_):    
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR_)):
        path = os.path.join(TEXT_DATA_DIR_, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
    
            labels_index[name] = label_id
            #print "labels_index",labels_index
            #print "name",name
            #print "label_id: ",label_id
            for fname in sorted(os.listdir(path)):
                #print "fname: ",fname
                fname_=fname.replace(".txt","")
                if fname_.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='utf8')
                    texts.append(f.read())
                    f.close()
                    #print "####### label_id: ",label_id 
                    labels.append(label_id)
    print "labels: ",labels
    print('Found %s texts.' % len(texts))
    
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    #print sequences
    word_index = tokenizer.word_index
    #print word_index
    print('Found %s unique tokens.' % len(word_index))
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    
    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    return data,labels,word_index,labels_index

data,labels,word_index,labels_index=get_data(TEXT_DATA_DIR)
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])


# x_train = data
# y_train = labels
# 
# x_val = data
# y_val = labels


x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


if mode=='s' :

    if os.path.isfile('modelsave.h5'):
        v=raw_input("trovato un modello vuoi davvero sovrascriverlo? Y/N: ")
        if not v =='Y':
            sys.exit(0)
    print('Preparing embedding matrix.')
    
    # prepare embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            print "Max nb words arrived!!!!!"
            continue
        #print word
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    
    
    
    
    print('Training model.')
    
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)
    
    model = Model(sequence_input, preds)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='rmsprop',
#                   metrics=['acc'])


#     model.compile(loss='mse',
#                   optimizer='rmsprop',
#                   metrics=['acc'])

    model.compile(loss=f1,
                  optimizer='rmsprop',
                  metrics=['acc'])


if mode=='l':
    print "LOAD MODEL..."
    model=load_model('modelsave.h5')


model.summary()
# happy learning!



if  mode=='s':
    
#     model.fit(x_train, y_train, validation_data=(x_val,y_val),
#               nb_epoch=10, batch_size=128)

    model.fit(x_train, y_train, validation_split=VALIDATION_SPLIT,
              nb_epoch=1, batch_size=128)


    model.save("modelsave.h5")


score, acc = model.evaluate(x_val, y_val,
                            batch_size=128)
        
print score,acc
prediction= (model.predict(x_val, verbose=2))

#print prediction

# soglia=0.5
# 
# for p in prediction:
#     print type(p)
#     print p
#     print p[0]
#     if p[0]>soglia:
#         classe=1
#     else:
#         classe=0
#         
#         
#    if classe==1 and 
confusion_matrix(prediction, y_val)

        #print zip(prediction,self.idTest)

        
#         print('Test score:', score)
#         print('Test accuracy:', acc)  







