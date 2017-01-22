
'''
Created on 03/ott/2016

@author: fabrizio
'''
from Config import  Config 
import os
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
        self.mappingID_CLASS =C.Datapath+os.sep+C.FileClassMapping
        self.w2vEmbeddingFileName = C.w2vEmbedding
        self.force=False
        self.nocontexwordsFlag=False
        
        
        