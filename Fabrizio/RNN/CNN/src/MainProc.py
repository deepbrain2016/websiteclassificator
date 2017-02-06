
import time

from MyDatasets import MyDatasets
from MyModels import MyModels
from keras.callbacks import EarlyStopping,ModelCheckpoint
import sys,os,numpy


def f1(y_true, y_pred):
    
    SUM=((y_pred-y_true)**2).sum(keepdims=True)   
    
    PROD=((y_pred*y_true)).sum(keepdims=True)   
    
    loss=-1/(1+(SUM/(2*(PROD+0.00001))))

    return loss

def f2(y_true, y_pred):
    

    
    TP=((y_pred*y_true)).sum(keepdims=True) 
    TN=(((1-y_pred)*(1-y_true))).sum(keepdims=True) 
    FP=(((y_pred)*(1-y_true))).sum(keepdims=True) 
    FN=(((1-y_pred)*(y_true))).sum(keepdims=True) 
    
    
    loss=-(TP/(0.001+TP+FN))+(TN/(0.001+TN+FP))

    return loss



def mse2(y_true, y_pred):
    

    
    loss=((1+y_true*(4-1))*(y_pred-y_true)**2).sum(keepdims=True) 
    #TN=(((1-y_pred)*(1-y_true))).sum(keepdims=True) 
    #FP=(((y_pred)*(1-y_true))).sum(keepdims=True) 
    #FN=(((1-y_pred)*(y_true))).sum(keepdims=True) 
    
    
    
    
    #loss=-(TP/(0.001+TP+FN))+(TN/(0.001+TN+FP))

    return loss




def confusion_matrix(prediction,Y_test):
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






parametri=sys.argv[1:]
print 'Number of arguments:', len(parametri), 'arguments.'
print 'Argument List:', str(parametri)




NomeRun=str(parametri[0])
TipoFile=str(parametri[1])
balance=str(parametri[2])
TypeNormalization=str(parametri[3])
ValidationSplit=float(parametri[4])
TypeNNet=str(parametri[5])

LossFunction=str(parametri[6])
LossFunctionString=LossFunction
if LossFunction=='f1':
    LossFunction= f1
if LossFunction=='f2':
    LossFunction= f2
if LossFunction=='mse2':
    LossFunction= mse2
    
Optimization=str(parametri[7])
BatchSize=int(parametri[8])
Patience=int(parametri[9])
Epocs=100
FormatModelSaved="../Run"+os.sep+NomeRun+os.sep+TypeNNet+"_saved.{epoch:02d}-{val_loss:.2f}.hdf5"


NClasses=MyModels.getNClasses(TypeNNet)
inputDIM=MyModels.getInputDIM(TypeNNet)
NetInDim=MyModels.getNetInDim(TypeNNet)

print ('#inizializza il seed del generatore dei numeri casuali sistema')
seed_random=int(time.time())
#seed_random=1474348309 #to run the best run ever

seed_random=1486100577
print "seed_random",seed_random


numpy.random.seed(seed_random)

print "#############  SEZIONE DATASETS #################"
DATA=MyDatasets(TipoFile,NetInDim,NClasses,inputDIM,balance,TypeNormalization,ValidationSplit,seed_random)
Ytrain,Xtrain,Ytest,Xtest=DATA.get()

#DATA.writeFiles(NomeRun)

print ("Xtrain.shape",Xtrain.shape)
print ("Ytrain.shape",Ytrain.shape)
print ("Xtest.shape",Xtest.shape)
print ("Ytest.shape",Ytest.shape)

print "#############  FINE SEZIONE  #################"


model=MyModels.build(TypeNNet)

print  "Compile model"
print "Le possibili loss function sono: \n\t mse,mae,mape,msle,squared_hinge,hinge,binary_crossentropy, \n\t categorical_crossentropy,sparse_categorical_crossentropy,\n\tkullback_leibler_divergence,poisson,cosine_proximity,f1,f2,mse2"
print "Selezionata: ",LossFunction
print "Le possibili optimization sono: \n\t SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam"
print "Selezionata: ",Optimization


model.compile(loss=LossFunction, optimizer=Optimization, metrics=['accuracy'])

print "Fit the model.."

early_stopping = EarlyStopping(monitor='val_loss', patience=Patience   )
checkPoint=ModelCheckpoint(FormatModelSaved, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')


history=model.fit(Xtrain, Ytrain, nb_epoch=Epocs, batch_size=BatchSize,validation_split=ValidationSplit, callbacks=[early_stopping,checkPoint],verbose=1)

prediction= (model.predict_classes(Xtest, verbose=2))



print "################# CONFUSION MATRIX SU BEST TRAINIG #################"
TP,TN,FP,FN,F1,PREC,REC,ACC =confusion_matrix(prediction,Ytest)


outTable=str(NomeRun)+" "+str(TipoFile)+" "+str(balance)+" "+str(TypeNormalization)+" "+str(ValidationSplit)+" "+str(TypeNNet)+" "+LossFunctionString+" "+str(Optimization)+" "+str(BatchSize)+" "+str(TP)+" "+str(TN)+" "+str(FP)+" "+str(FN)+" "+str(F1)+" "+str(PREC)+" "+str(REC)+" "+str(ACC)+" "+str(seed_random)+" "+str(Patience)+"\n"
TableFile=open("RunTable.csv",'a')
TableFile.write(outTable)
TableFile.close()

