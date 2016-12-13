'''
Created on 20/ott/2016

@author: fabrizio
'''
import unicodedata


class ItalianDict(object):

    
    
    
    def strip_accents(self,s):
        return unicodedata.normalize('NFD', s.decode('utf-8')).encode('ascii','ignore')
        #return  ''.join(c for c in unicodedata.normalize('NFD', s.decode('utf-8'))
        #      if unicodedata.category(c) != 'Mn')


    def CaricaDict(self):
        #NomeFileItaliano="/home/fabrizio/DeepLearning/RNN/ContenutiDATA/dizionario_italiano.csv"
        NomeFileItaliano="/home/fabrizio/ECLIPSE_PYTHON/RNN/ContenutiDATA/dizionarioItaEng.txt"
        FileItaliano=open(NomeFileItaliano,'r')
        print "\tCarica Dizionario ...",NomeFileItaliano
        for parola in FileItaliano:
            parolaDizionario=self.strip_accents(parola.strip("\r\n").lower())
            self.dizionario.add(parolaDizionario)
            #print parolaDizionario
        print "\tparole presenti nel dizionario :",len(self.dizionario)
        print "\tCarica Dizionario Fine"
        #print self.dizionario
        
    def isItalian(self,word):



        word_withoutAccent=self.strip_accents(word)
        #print word_withoutAccent
        a= word_withoutAccent in self.dizionario
        return a

       
    def filter(self,string):
        wordsInString=string.replace("/n","").replace("/r","").lower().split(" ")
        wordsFiltered=[]
        noword=set()
        for word in wordsInString:
            if self.isItalian(word):
                wordsFiltered.append(word)
            else:
                #print "no italian english filtred:",word
                noword.add(word)
        nowordFile=open(self.nowordFileName,'w')
        nowordFile.write( str(noword))
        
        return ' '.join(wordsFiltered)
         

    def __init__(self):
        '''
        Constructor
        '''
        self.dizionario=set()
        self.CaricaDict()
        self.nowordFileName="./nowordFile.txt"
        
# print "Start"
# ID=ItalianDict()
# while(True):
#     testo = input("?: ")
#     print ID.filter(testo)
# 
# print "End"
# 
