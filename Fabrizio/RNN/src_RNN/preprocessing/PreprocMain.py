'''
Created on 05/ott/2016

@author: fabrizio
'''

'''
Created on 03/ott/2016

@author: fabrizio
'''

import sys   
from PreprocessaContenuti import PreprocessaContenuti 
#from PreprocParameters  import PreprocParameters

def process():
    
    #Corpus="Contenuti.txt"
    Corpus="Contenuti100.txt"
    PPC=PreprocessaContenuti(Corpus)
    print "last output file: ",PPC.last_outputfile
    #PPC=PreprocessaContenuti("logcont.txt")
    PPC.filtraDizionarioItaliano()
    print "last output file: ",PPC.last_outputfile
    #sys.exit(0)
    
    PPC.seleziona_righe_pochi_caratteri(100001) #circa 80%
    print "last output file: ",PPC.last_outputfile
    #PPC.conta_righe_lastFile()
    #sys.exit(0)
    ##PreprocessaContenuti.estraiContex(listaParole,4)
    
    PPC.filtraContestoOntologia(5)
    print "last output file: ",PPC.last_outputfile
    #PPC.conta_righe_lastFile()
    
    encodincW2C=True
    if (encodincW2C==True):
        PPC.w2v()
        
    else:
        PPC.encoding()


    
    
    
    


print"Start Main"
process()
print"End Main"

    
    
    

