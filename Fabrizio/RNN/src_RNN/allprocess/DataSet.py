'''
Created on 03/ott/2016

@author: fabrizio
'''

from  Parameters import * 
from keras.datasets import imdb
from keras.preprocessing import sequence
import pickle,numpy,sys


class DataSet(object):
    '''
    classdocs
    '''

    def load_data_contex(self,nomefile):
        
        reader = pickle.load(open(nomefile, 'rb'))


        id=[]
        target=[]
        data=[]
        

        i_row=0
        for row in reader:
            #print row
            id.append(row[0])
            target.append(row[1])
            data.append(row[2])
            if (i_row==1000000):
                break
            i_row=i_row+1
            
        np_data=numpy.array(data)
        #print np_data
        print "shape",np_data.shape


        row_split=int(len(target)*(1-self.P.test_split))

        id_train=id[:row_split]
        X_train=data[:row_split]
        y_train=target[:row_split]

        id_test=id[row_split:]
        X_test=data[row_split:]
        y_test=target[row_split:]
        


        print "max_features: ",self.max_features
        print "max_site_length: ",self.max_site_length        
        return (id_train,X_train,y_train),(id_test,X_test,y_test)

        
        


    def sample_train(self, id_sample):
        print self.dataSet[0][0][id_sample]
        print self.dataSet[0][1][id_sample]

    def sample_test(self, id_sample):
        print self.dataSet[1][0][id_sample]
        print self.dataSet[1][1][id_sample]


    def load_data(self,tipoDataSet):
                print('Loading data...')
                if (tipoDataSet=='imdb'):
                    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=self.max_features)
                    '''debug'''
#                     X_train=X_train[0:100]
#                     y_train=y_train[0:100]
# 
#                     X_test=X_test[0:100]
#                     y_test=y_test[0:100]
                    ''''''

                    print ('dimensione_del_vocabolario:'), self.max_features
                    
                if (tipoDataSet=='contex3'):
                    (id_train,X_train,y_train),(id_test,X_test,y_test) = self.load_data_contex(self.P.contex3namefile)
                    print ('dimensione_del_vocabolario:'), self.max_features
                if (tipoDataSet=='preproc'):
                    (id_train,X_train,y_train),(id_test,X_test,y_test) = self.load_data_contex(self.P.preproc)
                    print ('dimensione_del_vocabolario:'), self.max_features                    
                
                if (tipoDataSet=='contexW2V'):
                    (id_train,X_train,y_train),(id_test,X_test,y_test) = self.load_data_contex(self.P.contexW2Vnamefile)
                    print ('dimensione_del_vocabolario:'), self.max_features                
                
                
                    
                print(len(X_train), 'train sequences')
                print(len(X_test), 'test sequences')
                return (X_train, y_train), (X_test, y_test) 


    def __init__(self):

        P=Parameters()
        self.P=P
        self.max_features=P.max_features
        self.max_site_length=P.max_site_length
        

        tipoDataSet=P.tipoDataSet
        

        (X_train, y_train), (X_test, y_test) =self.load_data(tipoDataSet)      
        #print X_train
        
        print('Pad sequences (samples x time   maxlen:)',self.max_site_length)
        print len(X_train[0][0])
        print (X_train[0][0])
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_site_length,dtype="float32")
        print len(X_train[0][0])
        print (X_train[0][998])
        
        print len(X_test[0][0])
        print (X_test[0][0])
        X_test = sequence.pad_sequences(X_test, maxlen=self.max_site_length,dtype="float32")
        print len(X_test[0][0])
        print (X_test[0][998])
        
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
        
        self.dataSet= (X_train, y_train), (X_test, y_test)  
        
        