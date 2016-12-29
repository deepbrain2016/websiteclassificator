'''
Created on 05/ott/2016

@author: fabrizio
'''
import unicodedata

import sys,os,re,pickle
import os.path
from PreprocParameters  import PreprocParameters
from ItalianDict import ItalianDict
from OntologiaEC import OntologiaEC
from gensim.models import word2vec


class PreprocessaContenuti(object):
    '''
    classdocs
    '''
        

    #@staticmethod
    
        
        

    def EcommerceDict(self):
        P=PreprocParameters()
        nomeFileTargetEcommerce=P.mappingID_CLASS
        DictEC={}

        fileTargetEcommerce=open(nomeFileTargetEcommerce,'r')
        for row in fileTargetEcommerce:

            row_split=row.split(',')
            cod=row_split[0]
            EC=row_split[2]

            DictEC[cod]=EC
        return DictEC
    

    
    def encoding(self):
        
        
        def Ecommerce_dict():
            P=PreprocParameters()
            nomeFileTargetEcommerce=P.WorkDir+"/"+"DATI_ictgrezzi2015.csv"
            DictEC={}
    
            fileTargetEcommerce=open(nomeFileTargetEcommerce,'r')
            for row in fileTargetEcommerce:

                row_split=row.split(',')
                cod=row_split[0]
                EC=row_split[2]

                DictEC[cod]=EC
            return DictEC
        
        
        
        print ("encoding")
        P=PreprocParameters()
        nomeinputfile=self.last_outputfile
        input_file=P.WorkDir+os.sep+nomeinputfile
        
        suffix="encoding_"
        suffix_mapping="mapping_"
        suffix_freq="freq_"
        postfix=".pkl"
        
        nome_output_file=P.WorkDir+os.sep+suffix+nomeinputfile+postfix       
        nome_output_file_mapping=P.WorkDir+os.sep+suffix_mapping+nomeinputfile 
        nome_output_file_freq=P.WorkDir+os.sep+suffix_freq+nomeinputfile
        
        if os.path.isfile(nome_output_file) and (not self.force):
        #if os.path.isfile(nome_output_file) :
            print "FILE encoding: "+ nome_output_file+" esiste"
            self.last_outputfile=suffix+input_file
            return 
        
        self.force=True

        
        to_encode =open(input_file,'r')
        

        
        
        EC_Dict=Ecommerce_dict()
        
        classe={}
        classe['1']=1
        classe['2']=0
        
        
        
        
        
        file_=open(input_file,'r')
        
        SetWords=set()
        conteggioParole={}
        for line in file_:
            AllWordsLine=filter(None, re.split("[\n\t ]+", line)[1:])
        

            for  word_in_AllWordsLine in AllWordsLine:
                        try:
                            conteggioParole[word_in_AllWordsLine]=conteggioParole[word_in_AllWordsLine]+1
                        except:
                            conteggioParole[word_in_AllWordsLine]=1
                        SetWords.add(word_in_AllWordsLine)

        mapping={}
        i=0
        
        FreqFile=open(nome_output_file_freq,'w')
        for k in conteggioParole.iterkeys():
            out_conteggio= str(k)+";"+str(conteggioParole[k])+"\n"
            FreqFile.write(out_conteggio)
        FreqFile.close()
            

        for w in SetWords:
                mapping[w]=i
                i=i+1

        print "parole individuate nel Corpus: "+str(i)
        
        EncodingFile=open(nome_output_file_mapping,'w')
        EncodingFile.write(str(mapping))

        file_.close()
        

        
        inte=True      
        Corpus_coded=[]

        for row in to_encode:
            row=row+" "
        
            row_split=row.split("\t")
            
            COD_UNITA=row_split[0]
            
            try:
                row=row_split[1]
            except:
                print "except", row
                
            if inte == True :
                inte=False
                continue
            
            encode_row=[]
            words=row.split(" ")
            SetWords=set(words)
            n_word_inserite=0
            for word in words:

                try:

                    w=word.replace("\n","")
                
                    code=mapping[w]
                except:

                    continue
                encode_row.append(code)
                n_word_inserite=n_word_inserite+1
                

            
            try:    
                EC_FLAG=EC_Dict[COD_UNITA]
        
            
                if EC_FLAG=='':
                    continue
                EC_CLASSE = classe[EC_Dict[COD_UNITA]]
                Corpus_coded.append([COD_UNITA,EC_CLASSE,encode_row])
        
            except:
                print "except",COD_UNITA

        ##print EC_Dict
        file_encode=open(nome_output_file,'wb')
        file_txt_encode=open(nome_output_file+".txt",'w')
        pickle.dump(Corpus_coded, file_encode)
        
        for row in Corpus_coded:
            file_txt_encode.write(str(row)+'\n')
            
        file_encode.close()    
        file_txt_encode.close()    
            
        
        
        

    
    
    
    
    def seleziona_righe_pochi_caratteri(self,CaratteriSitoMassimi):
        print ("Il metodo seleziona le righe del testo con meno caratteri \
        di quelli specificati. Le rimanenti vengono scartate ")
        
        nomefile=self.last_outputfile
        
        P=PreprocParameters()
        input_file=P.WorkDir+os.sep+nomefile
        suffix="PochiCaratteri"+str(CaratteriSitoMassimi)
        nome_output_file=P.WorkDir+os.sep+suffix+nomefile
        
        
        
        
        if os.path.isfile(nome_output_file) and (not self.force):
        #if os.path.isfile(nome_output_file):
            print "FILE: "+ nome_output_file+" esiste"
            self.last_outputfile=suffix+nomefile
            return 
        self.force=True
        output_file=open (nome_output_file,'w')
        
        Contenuti =open(input_file,'r')
        print "file di input "+input_file+"...reading"
        lineeScritte=0
        lineeTot=0
        
        for linea in Contenuti:
            try:
                linea[CaratteriSitoMassimi]

            except:
                #print "Riga con meno di: "+str(CaratteriSitoMassimi)+"Caratteri"
                output_file.write( linea.replace('\n',"").replace('\r',"")+"\n" )
                lineeScritte=lineeScritte+1
                pass
            lineeTot=lineeTot+1
            
            if (lineeTot%100==0):
                print "lineeScritte: ",lineeScritte
                print "lineeTot: ",lineeTot
        
        output_file.close()
        self.last_outputfile=suffix+nomefile
        print "file: "+self.last_outputfile+"... write" 
        return

    def conta_righe_lastFile(self):
        P=PreprocParameters()
        file_to_count=open (P.WorkDir+os.sep+self.last_outputfile,'r')
        n_righe= len(file_to_count.readlines())
        print ("Numero di righe selezionate: ",n_righe)
        return n_righe


    def w2v(self):


        def getw2v(word,model):
            try:
                if word=="#":
                    word="fine"
                word_without_accents=unicodedata.normalize('NFD', word.decode('utf-8')).encode('ascii','ignore')
                vec=model[word_without_accents]

            except:
                #print "exception",word
                #print word_without_accents
                vec=model['mare']
            return vec


        
        P=PreprocParameters()
        suffix="W2V"
        postfix=".pkl"

        print "Word2Vec encoding is started...."
        nome_input_file=P.WorkDir+os.sep+self.last_outputfile
        nome_output_file=P.WorkDir+os.sep+suffix+self.last_outputfile+postfix

        if os.path.isfile(nome_output_file) and (not self.force):
        #if os.path.isfile(nome_output_file):

            print "FILE: "+ nome_output_file+" esiste"
            self.last_outputfile=suffix+self.last_outputfile
            return 
        self.force=True
        
        fileInput4W2V=open(nome_input_file,'r')
        
        fileOutputW2V=open(nome_output_file,'w')        
        ################# start proc ###################
        
        EC_Dict=self.EcommerceDict()
        
        
        w2vEmbeddingFileName=P.w2vEmbeddingFileName
        print "File w2v input:",w2vEmbeddingFileName
        
        model= word2vec.Word2Vec.load_word2vec_format(w2vEmbeddingFileName, binary=True)
        ''' per ogni riga, per ogni parola del contenuto scrapata
        trova la rappresentazione vettoriale di word2vec 
        salvala in una struttura (n_siti,dim_spazio_vet)'''
        
        vecCorpus=[]
        for line in fileInput4W2V:
            SiteIDYX=[]
            lineSplit= line.split('\t')
            try:    
                ID=lineSplit[0]
                site=lineSplit[1]
                vecLine=[]
                siteSplit=site.split(' ')
                for word in siteSplit:
                    vecWord= getw2v(word,model)
                    vecLine.append(vecWord)
                print ID,EC_Dict[ID]
                Y = self.classe[EC_Dict[ID]]
                SiteIDYX.append(ID)
                SiteIDYX.append(Y)
                SiteIDYX.append(vecLine)
                vecCorpus.append(SiteIDYX)
            except:
                print 
                print "exception ID not found into EC_DICT",lineSplit
        ################# stop proc ####################
        print "dim siti: ",len(vecCorpus)
        print "dim elementi: ",len(vecCorpus[0])
        print vecCorpus[0] 
        print "dim parole sito: ",len(vecCorpus[0][0]) 
        print "dim spazio vettoriale: ",len(vecCorpus[0][2][0])
        
        
        pickle.dump(vecCorpus, fileOutputW2V)

        
        fileInput4W2V.close()
        fileOutputW2V.close()    
            
        
        
        self.last_outputfile=suffix+self.last_outputfile
        return
                      


    def filtraDizionarioItaliano(self):
        print "filtraDizionario"
        P=PreprocParameters()
        suffix="FilterItaEngNew"
        
        nome_input_file=P.WorkDir+os.sep+self.last_outputfile
        nome_output_file=P.WorkDir+os.sep+suffix+self.last_outputfile
        
        
        if os.path.isfile(nome_output_file) and (not self.force):     
        #if os.path.isfile(nome_output_file) :
            print "FILE: "+ nome_output_file+" esiste"
            self.last_outputfile=suffix+self.last_outputfile
            print "filtraDizionario end"
            return 
        self.force=True
        filetofilter=open(nome_input_file,'r')
        filefiltered=open(nome_output_file,'w')
        ItalianDiz=ItalianDict()
        
        #ItalianDict.isItalian()
        n=0
        for row in filetofilter:
            n+=1
            if n%10==0:
                print "n rows filtred: ",n
            [ID,string]=row.split("\t")
            stringFiltered=ItalianDiz.filter(string)
            newRow=ID+"\t"+stringFiltered
            filefiltered.write(newRow+"\n")
        
        
        filefiltered.close()
        filetofilter.close()    
            
        
        
        self.last_outputfile=suffix+self.last_outputfile
        print "filtraDizionario end"
        return



    def filtraContestoOntologia(self,ParoleContesto):
        
        def checkSplitID(ID):
            try:
                if int(ID)==0:
                    print "ID valore anomalo"
                    return False
            except:
                    print "ID valore anomalo"
                    return False
            return True              
        
        print "Filtra Ontologia...."
        P=PreprocParameters()
        suffix="Contex"+str(ParoleContesto)
        
        nome_input_file=P.WorkDir+os.sep+self.last_outputfile
        nome_output_file=P.WorkDir+os.sep+suffix+self.last_outputfile
        
        
        if os.path.isfile(nome_output_file) and (not self.force):
        #if os.path.isfile(nome_output_file) :
            print "FILE: "+ nome_output_file+" esiste"
            self.last_outputfile=suffix+self.last_outputfile
            return 
        self.force=True
        filetofilter=open(nome_input_file,'r')
        filefiltered=open(nome_output_file,'w')
        Ontologia=OntologiaEC()
        
        for row in filetofilter:
            [ID,string]=row.split("\t")
            if checkSplitID(ID)==True:
                parole=Ontologia.paroleContestoOntologia(string,ParoleContesto)
                if len(parole)!=0:
                    outrow= " # ".join(parole)
                    outrow=outrow.replace("\n","")
                    outrow=outrow.replace("\r","")
                else:
                    outrow="nocontexwords"
                stringOut= ID+"\t"+outrow
                filefiltered.write(stringOut+"\n")   
            else:
                print "Riga con ID non numerico"
                row=row.replace("\n","")
                row=row.replace("\r","")
                filefiltered.write(row+"\n")
                 
            
                
        
        


    def __init__(self,CorpusFile):
        '''
        Constructor
        '''
        P=PreprocParameters()

        print "init PreprocessaContenuti"
        self.last_outputfile=CorpusFile
        
        self.force=P.force
        
        self.classe={}
        self.classe['1']=1
        self.classe['2']=0
    
        
        
        
        print "init PreprocessaContenuti .... end"