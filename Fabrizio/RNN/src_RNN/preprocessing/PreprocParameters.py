
'''
Created on 03/ott/2016

@author: fabrizio
'''
from Config import  Config 
class PreprocParameters(object):
    '''
    classdocs
    '''
    
        


    def __init__(self):
        '''
        Constructor
        '''
        C=Config()
        self.WorkDir = C.Datapath
        self.w2vEmbeddingFileName = "/home/fabrizio/ECLIPSE_PYTHON/RNN/word2vect_stuff/contenuti100.model.bin"
        self.force=False
        
        