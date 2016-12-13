'''
Created on 03/ott/2016

@author: fabrizio
'''

class Parameters(object):
    '''
    classdocs
    '''
    
    


    def __init__(self):
        '''
        Constructor
        '''
   
           #self.fileInput="prova.pkl"
        self.tipoDataSet="contex3"#"imdb"
        self.tipoDataSet="contexW2V"#"imdb"
        #self.tipoDataSet="imdb"  

        
        self.contexW2Vnamefile="/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/W2VContex5PochiCaratteri100001FilterItaEngNewContenuti100.txt.pkl"
        
        self.contex3namefile="/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/encoding_Contex5PochiCaratteri1000001FilterItaEngNewContenuti.txt.pkl"
        self.preproc=""

    
        
        self.embedding_vecor_length=0
#         self.max_features=5000
#         self.max_site_length=500
        
        self.max_features=0
        self.max_site_length=1000

        self.batch_size = 64
        self.epoch = 3
        


        
        self.ValidationSplit=0.1
        self.test_split=0.3
        
        
        self.TypeNN="LSTM"
        self.TypeNN="CNNLSTM"
        self.TypeNN="V2WLSTM"
        self.TypeNN="V2WCNNLSTM"
        
        self.N_LSTM=100
        
  
        #self.preproc="/home/fabrizio/DeepLearning/RNN/ContenutiDATA/encoding_PochiCaratteri1500Contenuti.txt.pkl"
        
