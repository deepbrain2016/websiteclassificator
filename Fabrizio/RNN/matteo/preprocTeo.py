class Performance():
    
    def __init__(self,ElencoCatPrinc,CatSec2CatPrinc):
        self.PerformaceTable={}
        self.fout=open("./outMatteo.csv",'w')
        for CatPrinc in ElencoCatPrinc:
            self.PerformaceTable[CatPrinc]=[]
    def setBambino(self,bamb):
        self.bamb=bamb
    def getTable(self):
        return self.PerformaceTable
    def PrintTable(self):
        for i in self.PerformaceTable.keys():
            if len(self.PerformaceTable[i])==0 :
                continue
            perf= "##PRINT TABLE## BAMBINO:"+self.bamb+";"+str(i)+";"+str(self.PerformaceTable[i])
            print perf
    def nuovaPerformance(self,CatSec,value):
        CatPrinc=CatSec2CatPrinc[CatSec]
        self.PerformaceTable[CatPrinc].append([CatSec,value])
    def getCategoriaPrincipale(self,CatPrinc):
        return self.PerformaceTable[CatPrinc]    
    
    def uniquePerformanceSecondarie(self,groupby='MAX'):
        
        for i in self.PerformaceTable.keys():
            PerformanceSecondarieUniq={}
            elencoPerformanceSec=self.PerformaceTable[i]
            for PerfSec,value in  elencoPerformanceSec:
                try:
                    if value>PerformanceSecondarieUniq[PerfSec]:
                        PerformanceSecondarieUniq[PerfSec]=value
                except:
                    PerformanceSecondarieUniq[PerfSec]=value
            self.PerformaceTable[i]=PerformanceSecondarieUniq
                    
    def uniquePerformancePrimariaScore(self):
        for i in self.PerformaceTable.keys():
            PerformanceSecondarieUniq=self.PerformaceTable[i]  
            vTot=0
            for v in PerformanceSecondarieUniq.values():
                #print i,v
                vTot=vTot+float(v.replace(',','.'))
            if len(PerformanceSecondarieUniq)<>0:
                vTot=float(vTot)/(len(PerformanceSecondarieUniq))
                #print "### SCORE:",i,vTot
            self.PerformaceTable[i]=[vTot]
            
                     
                    



fileInputName="//home//fabrizio//Scrivania//MatteoData//autismo.csv"

inputFile=open (fileInputName,'r')

ElencoCatPrinc=set()
ElencoCatSec=set()
ElencoBambini=set()

CatSec2CatPrinc={}
for line in inputFile:
    lineSplit=line.split(";")
    CatPrinc=lineSplit[4]
    CatSec=lineSplit[5]
    ElencoCatPrinc.add(CatPrinc)
    ElencoCatSec.add(CatSec)
    CatSec2CatPrinc[CatSec]=CatPrinc
    
    Bambini=lineSplit[0]
    ElencoBambini.add(Bambini)

#print ElencoCatPrinc
#print len(ElencoCatPrinc)
#print len(ElencoCatSec)


# for sec in ElencoCatSec:
#     print CatSec2CatPrinc[sec],sec

    
    


#print ElencoBambini  

PerformanceBambino={}
for bambino in ElencoBambini:
    PerformanceBambino[bambino]=Performance(ElencoCatPrinc,CatSec2CatPrinc)
    PerformanceBambino[bambino].setBambino(bambino)



for bambino in ElencoBambini:
    PerformanceBambino[bambino].PrintTable()

inputFile=open (fileInputName,'r')

for line in inputFile:
    ls= line.split(";")
    Bambino=ls[0]
    CatPrinc=ls[4]
    CatSec=ls[5]
    value=ls[9]
    PerforfomanceDelBambinoOBJ=PerformanceBambino[Bambino]
    #print Bambino,value
    PerforfomanceDelBambinoOBJ.nuovaPerformance(CatSec,value)

head=True
for bambino in ElencoBambini:
    if head==True:
        row=""
        TABLE=PerformanceBambino[bambino].getTable()
        for CAT in TABLE.keys():
            row=row+";"+CAT
        head=False
        print "bambino;"+row.strip(";")
    pass


for bambino in ElencoBambini:
    row=""
    PerformanceBambino[bambino].uniquePerformanceSecondarie()
    #PerformanceBambino[bambino].PrintTable()
    PerformanceBambino[bambino].uniquePerformancePrimariaScore()
    #PerformanceBambino[bambino].PrintTable()
    TABLE=PerformanceBambino[bambino].getTable()
    for CAT in TABLE.keys():
        row=row+";"+str(TABLE[CAT][0])
    print bambino+";"+row.strip(";")


    
    #for CatPinc in encoCatPrinc:
    #     elencoDiPerformance=PerformanceBambino[bambino].getCategoriaPrincipale(CatPrinc)

 