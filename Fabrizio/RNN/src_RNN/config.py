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
        
        Config.read('../GlobalPar.cfg')
    
        
    
        #Config.add_section('Global')
        #config.set('Global', 'CorpusFilePath', ' /home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/Contenuti.txt')
    
    
    
        self.Corpus = Config.get('Global', 'CorpusFilePath')


