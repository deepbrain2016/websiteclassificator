'''
Created on 22/ott/2016

@author: fabrizio
'''
import os,ast,numpy as np
class RicostruzioneContenuti(object):
    '''
    classdocs
    '''


    def __init__(self, workDir,File_mapping,File_encoding):
        pathMapping=workDir+os.sep+File_mapping
        pathEncoding=workDir+os.sep+File_encoding
        DescrMapping=open(pathMapping,'r')
        DescrEncodig=open(pathEncoding,'r')
        
        for row in DescrMapping:
            #print row
            self.Mapping=ast.literal_eval(row)
            
        self.MappingInv={}
        for key in self.Mapping.iterkeys():
            val=self.Mapping[key]
            self.MappingInv[val]=key
        

            
        
        DescrMapping.close()
        
        outFile=open(workDir+os.sep+"RecostructioOut.txt",'w')
        #outFileEC=open(workDir+os.sep+"RecostructioOutEC.txt",'w')
        listEC=[]
        listNoEC=[]

        for row in DescrEncodig:
            #print row
            row=ast.literal_eval(row)
            
            cont_row=row[2]
            codED=row[1]
            codUnita=row[0]
            outFile.write(str(codUnita)+";"+str(codED)+";")
            listRowEC=[]
            listRowNoEC=[]

            if (codED==1):
                listRowEC.append(codUnita)
            else :
                listRowNoEC.append(codUnita)
                
            for cod in cont_row:
                word= self.MappingInv[cod]
                outFile.write(word+" ")
                if (codED==1):
                    listRowEC.append(word)
                else:
                    listRowNoEC.append(word)
                    
            outFile.write("\n")
            
            if (codED==1):
                listEC.append(listRowEC)
            else :
                listNoEC.append(listRowNoEC)
                
                
        DescrMapping.close()
        outFile.close()
        
        print "Fine Ricostruzione Encoding"
        print range(len(listNoEC))
        indexNpNoEC=np.array(range(len(listNoEC)))
        np.random.shuffle(indexNpNoEC)
        indexEstrai20NoEC= indexNpNoEC[:20]
        estrai20NoEC=[]
        for index in indexEstrai20NoEC:
            estrai20NoEC.append(listNoEC[index])
        
        
        print "len estrai20NoEC: ",len(estrai20NoEC)
        print estrai20NoEC
        
        
        npEC=np.asarray(listEC)
        np.random.shuffle(npEC)
        estrai20= npEC[:20]
        
        print "len estrai20EC: ",len(estrai20)
        
        
        #print estrai20.shape()
        #np.savetxt(workDir+os.sep+"estrai20.txt",estrai20, fmt='%.10000s')
        
        estrTxt=open(workDir+os.sep+"estrai20ItaEng.txt",'w')
        for row in estrai20:
            string=' '.join(row)
            estrTxt.write(string+'\n')
        estrTxt.close()
        
        estrNoECTxt=open(workDir+os.sep+"estrai20NoECItaEng.txt",'w')
        for row in estrai20NoEC:
            string=' '.join(row)
            estrNoECTxt.write(string+'\n')
        estrNoECTxt.close()
        
        
        
        

RicostruzioneContenuti('/home/fabrizio/ECLIPSE_PYTHON/RNN/ContenutiDATA',
                       'mapping_Contex5PochiCaratteri1000000FilterItaEngNewContenuti.txt',
                       'encoding_Contex5PochiCaratteri1000000FilterItaEngNewContenuti.txt.pkl.txt')

        
        
