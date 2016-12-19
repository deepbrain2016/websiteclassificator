import ConfigParser


class Config(object):
    
    
    def __init__(self):

        
        Config = ConfigParser.ConfigParser()
#         cfgfile = open("../GlobalPar.cfg",'w')
# 
#         Config.add_section('Global')
#         Config.set('Global','CorpusFilePath',' /home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/Contenuti.txt')
#         Config.write(cfgfile)
#         cfgfile.close()
        
        Config.read('../../GlobalPar.cfg')
    

    
        self.Datapath = Config.get('PreProcessing', 'Datapath')
        self.Corpus = Config.get('PreProcessing', 'corpusfile')
        self.Dictonary = Config.get('PreProcessing', 'Dictonary')
        self.w2vEmbedding = Config.get('Embedding', 'W2VAbsFile')


