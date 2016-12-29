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
        #self.tipoDataSet="contexW2V"#"imdb"
        #self.tipoDataSet="imdb"  

        
        self.contexW2Vnamefile="/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/W2VContex5PochiCaratteri100001FilterItaEngNewContenuti100.txt.pkl"
        
        #self.contex3namefile="/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/encoding_Contex5PochiCaratteri1000001FilterItaEngNewContenuti.txt.pkl"
        self.contex3namefile="/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/encoding_PochiCaratteri10000FilterItaEngNewDataSetPoeFrostCODUPPER.csv.pkl"
        self.contex3namefile="/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/encoding_PochiCaratteri10000FilterItaEngNewContenutiCut1000Char.txt.pkl"
        self.contex3namefile="/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/encoding_Contex5PochiCaratteri10000FilterItaEngNewContenutiCut1000Char.txt.pkl"
        
        
        self.preproc=""

    
        
        self.embedding_vecor_length=60
#         self.max_features=5000
#         self.max_site_length=500
        
        #self.max_features=1100 #poe frost dataset
        self.max_features=36534
        self.max_site_length=100

        self.batch_size = 10
        self.epoch = 2
        

        
        self.ValidationSplit=0.01
        self.test_split=0.2
        
        
        self.TypeNN="LSTM"
        #self.TypeNN="CNNLSTM"
        #self.TypeNN="V2WLSTM"
        #self.TypeNN="V2WCNNLSTM"
        
        self.N_LSTM=100
        
  
        #self.preproc="/home/fabrizio/DeepLearning/RNN/ContenutiDATA/encoding_PochiCaratteri1500Contenuti.txt.pkl"
        
