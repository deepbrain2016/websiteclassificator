'''
Created on 03/ott/2016

@author: fabrizio
'''

import numpy

class Performance(object):
    '''
    classdocs
    '''

    
    def confusion_matrix(self,TestID_Prediction,Y_test):
        NClasses=1
        TP=0
        TN=0
        FP=0
        FN=0
        i=0
        elencoTN=[]
        elencoTP=[]
        elencoFN=[]
        elencoFP=[]        
        
    
        for testID,y_p in TestID_Prediction:

            if NClasses==1:
                if y_p[0]==1 and Y_test[i]==1: 
                    TP=TP+1
                    elencoTP.append(testID)
                if y_p[0]==0 and Y_test[i]==0: 
                    TN=TN+1
                    elencoTN.append(testID) 
                if y_p[0]==0 and Y_test[i]==1: 
                    FN=FN+1 
                    elencoFN.append(testID)
                if y_p[0]==1 and Y_test[i]==0: 
                    FP=FP+1 
                    elencoFP.append(testID)
                
            if NClasses==2:            
                if y_p==1 and numpy.dot(Y_test[i],[0,1])==1: 
                    TP=TP+1
                    elencoTP.append(testID)
                if y_p==0 and numpy.dot(Y_test[i],[0,1])==0: 
                    TN=TN+1 
                    elencoTN.append(testID)
                if y_p==0 and numpy.dot(Y_test[i],[0,1])==1: 
                    FP=FP+1 
                    elencoFP.append(testID)
                if y_p==1 and numpy.dot(Y_test[i],[0,1])==0: 
                    FN=FN+1 
                    elencoFN.append(testID)
                
            i=i+1
        self.elencoTN=elencoTN
        self.elencoTP=elencoTP
        self.elencoFN=elencoFN
        self.elencoFP=elencoFP
    
        print "TP",TP
        print "TN",TN
        print "FP",FP
        print "FN",FN
    
        F1=2*float(TP)/(2*float(TP)+float(FN)+FP)
        ACC=(float(TP)+TN)/(TN+TP+FN+FP)
        try:
            PREC=float(TP)/(TP+FP)
        except:
            PREC=99999
        try:
            REC=float(TP)/(TP+FN)
        except:
            PREC=99999
            
    
        
    
        print "F1: "+str(F1)
        print "PREC: "+str(PREC)
        print "REC: "+str(REC)
        print "ACC: "+str(ACC)
        return TP,TN,FP,FN,F1,PREC,REC,ACC
    
    
    
    


    def __init__(self,score,acc,pred,y_real4pred,test_id):
        def CreaTestID_Prediction(test_id,pred):
#             prediction_testID={}
#             for k,v in zip(test_id,pred):
#                 prediction_testID[k]=v
#                 return prediction_testID
            return zip(test_id,pred)
        '''
        Constructor
        '''
        self.score=score
        self.acc=acc
        self.pred=pred
        self.y_real4pred=y_real4pred
        self.test_id=test_id
        self.testID_prediction=CreaTestID_Prediction(test_id,pred);
        
    def write(self):
        print "SCORE: ",self.score
        print "ACC: ",self.acc
        
        TP,TN,FP,FN,F1,PREC,REC,ACC=self.confusion_matrix(self.testID_prediction,self.y_real4pred)
        print self.elencoFN
        print "self.elencoFN"
        print self.elencoTN
        print "self.elencoTN"
        print self.elencoFP
        print "self.elencoFP"
        print self.elencoTP
        print "self.elencoTP"
        
#        filedata=open ("/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/DataSetPoeFrostCODUPPER.csv",'r')
        filedata=open ("/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/ContenutiCut10000Char.txt",'r')

        dictDatafile={}
        for line in filedata:
            (k,v)= line.split("\t")
            #print k
            #print v
            dictDatafile[k]=v
            
#        filedataw=open ("/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/DataSetPoeFrostCODout.csv",'w')
        filedataw=open ("/home/fabrizio/DEVPYTHON/RNN/ContenutiDATA/ContenutiCut10000Charout.txt",'w')
            
        for k in self.elencoFN:
            w= "FN;"+k+";"+dictDatafile[k]
            filedataw.write(w)
            
        for k in self.elencoFP:
            w= "FP;"+k+";"+dictDatafile[k]
            filedataw.write(w)
        for k in self.elencoTN:
            w= "TN,"+k+";"+dictDatafile[k]
            filedataw.write(w)
        for k in self.elencoTP:
            w= "TP;"+k+";"+dictDatafile[k]
            filedataw.write(w)            
            
            
        
        