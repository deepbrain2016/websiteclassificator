'''
Created on 04/gen/2017

@author: fabrizio
'''
classe={}
classe['1']=1
classe['2']=0


import random

def Ecommerce_dict():

    nomeFileTargetEcommerce="/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/DATI_ictgrezzi2015.csv"
    DictEC={}

    fileTargetEcommerce=open(nomeFileTargetEcommerce,'r')
    for row in fileTargetEcommerce:

        row_split=row.split(',')
        cod=row_split[0]
        EC=row_split[2]

        DictEC[cod]=EC
    return DictEC




EC_Dict=Ecommerce_dict()

def checkSplitID(ID):
    try:
        if int(ID)==0:
            print "ID valore anomalo"
            return False
    except:
            print "ID valore anomalo"
            return False
    return True     

if __name__ == '__main__':
    
        nome_input_file="/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/FilterItaEngNewContenutiCut10000Char.txt"
        nome_input_file="/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/Contex5PochiCaratteri10000FilterItaEngNewContenutiCut10000Char.txt"
        filetofilter=open(nome_input_file,'r')
        
        i=0
        for row in filetofilter:
            print i
	    #if i > 500 :
	    #	break
            [ID,string]=row.split("\t")
            string=string.replace("#","stop")
            if checkSplitID(ID)==True:
                print string
                try:
                    EC_CLASSE = classe[EC_Dict[ID]]
                except:
                    continue
                dir=None
                
                #if i<1000:
                #if random.randint(1,10000)<1000:
                if random.randint(1,10000)<-50:
                    if EC_CLASSE==0:
                        dir="EcommerceTestContex/EC0"
                        NomeFile=dir+"/"+ID+".txt"
                        f=open (NomeFile,"w")
                        f.write(string)
                        f.close
                    if EC_CLASSE==1:
                        dir="EcommerceTestContex/EC1"
                        NomeFile=dir+"/"+ID+"00000.txt"
                        f=open (NomeFile,"w")
                        f.write(string)
                        f.close
                else:
                
                    if EC_CLASSE==0:
                        dir="EcommerceContex/EC0"
                        NomeFile=dir+"/"+ID+".txt"
                        f=open (NomeFile,"w")
                        f.write(string)
                        f.close
                    if EC_CLASSE==1:
                        dir="EcommerceContex/EC1"
                        NomeFile=dir+"/"+ID+"00000.txt"
                        f=open (NomeFile,"w")
                        f.write(string)
                        f.close
                        #NomeFile=dir+"/"+ID+"00001.txt"
                        #f=open (NomeFile,"w")
                        #f.write(string)
                        #f.close
                        #NomeFile=dir+"/"+ID+"00002.txt"
                        #f=open (NomeFile,"w")
                        #f.write(string)
                        #f.close
                        #NomeFile=dir+"/"+ID+"00003.txt"
                        #f=open (NomeFile,"w")
                        #f.write(string)
                        #f.close
            i+=1
                
