'''
Created on 03/ott/2016

@author: fabrizio
'''

import numpy

class Performance(object):
    '''
    classdocs
    '''

    
    def confusion_matrix(self,prediction,Y_test):
        NClasses=1
        TP=0
        TN=0
        FP=0
        FN=0
        i=0
    
        for y_p in prediction:
            
            if NClasses==1:
                if y_p[0]==1 and Y_test[i]==1: 
                    TP=TP+1
                if y_p[0]==0 and Y_test[i]==0: 
                    TN=TN+1 
                if y_p[0]==0 and Y_test[i]==1: 
                    FN=FN+1 
                if y_p[0]==1 and Y_test[i]==0: 
                    FP=FP+1 
                
            if NClasses==2:            
                if y_p==1 and numpy.dot(Y_test[i],[0,1])==1: 
                    TP=TP+1
                if y_p==0 and numpy.dot(Y_test[i],[0,1])==0: 
                    TN=TN+1 
                if y_p==0 and numpy.dot(Y_test[i],[0,1])==1: 
                    FP=FP+1 
                if y_p==1 and numpy.dot(Y_test[i],[0,1])==0: 
                    FN=FN+1 
                
            i=i+1
    
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
    
    
    
    
    
    

    def __init__(self,score,acc,pred,y_real4pred):
        '''
        Constructor
        '''
        self.score=score
        self.acc=acc
        self.pred=pred
        self.y_real4pred=y_real4pred
        
    def write(self):
        print "SCORE: ",self.score
        print "ACC: ",self.acc
        
        TP,TN,FP,FN,F1,PREC,REC,ACC=self.confusion_matrix(self.pred,self.y_real4pred)
        
        

