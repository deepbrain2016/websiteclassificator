import numpy,os
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import EarlyStopping,ModelCheckpoint



def creaDataSet(list_test):
    fileInputName="//home//fabrizio//Scrivania//MatteoData//aut3.csv"
    fileInput=open(fileInputName,'r')
    
    def pulisci(X):
        i=0
        for v in X: 
            X[i]=v.replace('\n','')
            i+=1
        return X
    
    X=[]
    Y=[]
    for line in fileInput:
         Yi= line.split(";")[0]
         Xi= line.split(";")[1:]
         Xi= pulisci(Xi)
         X.append(Xi)
         Y.append(Yi)
    
    
    for b1 in range(11):
         for b2 in range(b1+1,11):
             print "b1,b2:",b1,b2
             A=X[b1]
             B=X[b2]
             s=0
             for i in range(len(A)):
                 s=s+(float(A[i])-float(B[i]))**2
             print "dist:",s
    
    
    
    
    fileInput=open(fileInputName,'r')
        
    X_train=[]
    Y_train=[]
    i=0
    for line in fileInput:
         if i in list_test:
            i+=1
            continue
         Yi= line.split(";")[0]
         Xi= line.split(";")[1:]
         Xi= pulisci(Xi)
         X_train.append(Xi)
         Y_train.append(Yi)
         i+=1
         
    fileInput=open(fileInputName,'r')
    
    X_test=[]
    Y_test=[]
    i=0
    for line in fileInput:
         if not i in list_test:
            i+=1
            continue
         Yi= line.split(";")[0]
         Xi= line.split(";")[1:]
         Xi= pulisci(Xi)
         X_test.append(Xi)
         Y_test.append(Yi)
         i+=1
    return X_test,Y_test,X_train,Y_train
    
    
    
 



def go(list_test,X_test,Y_test,X_train,Y_train):
    
   
    
    
    def confusion_matrix(prediction,Y_test):

        i=0
        sum=0
        for y_p in prediction:
            
            print y_p,Y_test[i]
            sum=sum+(float(y_p[0])-float(Y_test[i]))*(float(y_p[0])-float(Y_test[i]))
            i+=1
        print "###############################perf:",sum
    
    
    
    
    
    
    
    
    # 
    # # define base mode
    # def baseline_model():
    #     # create model
    #     model = Sequential()
    #     model.add(Dense(13, input_dim=142, init='normal', activation='relu'))
    #     model.add(Dense(1, init='normal'))
    #     # Compile model
    #     model.compile(loss='mean_squared_error', optimizer='adam')
    #     return model
    # 
    # # fix random seed for reproducibility
    # seed = 7
    # numpy.random.seed(seed)
    # # evaluate model with standardized dataset
    # estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=1000, batch_size=1, verbose=1)
    # 
    # 
    # kfold = KFold(n_splits=2, random_state=seed)
    # results = cross_val_score(estimator, X, Y, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    
    NomeRun='provaMLP'
    Epocs=500
    FormatModelSaved="_saved.{epoch:02d}-{val_loss:.2f}.hdf5"
    BatchSize=2
    
    #print  "Compile model"
    #print "Le possibili loss function sono: \n\t mse,mae,mape,msle,squared_hinge,hinge,binary_crossentropy, \n\t categorical_crossentropy,sparse_categorical_crossentropy,\n\tkullback_leibler_divergence,poisson,cosine_proximity,f1,f2,mse2"
    LossFunction='mse'
    Optimization='Adam'
    Patience=15
    ValidationSplit=0
    #rint "Selezionata: ",LossFunction
    #print "Le possibili optimization sono: \n\t SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam"
    #print "Selezionata: ",Optimization
    
    
    
    #print "Seleziona il modello mpl 142 12 8 1"
    model = Sequential()
    model.add(Dense(12, input_dim=142, init='uniform', activation='relu'))
    model.add(Dense(5, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='relu'))
    
    
    
    model.compile(loss=LossFunction, optimizer=Optimization, metrics=['accuracy'])
    
    #print "Fit the model.."
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=Patience   )
    checkPoint=ModelCheckpoint(FormatModelSaved, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
    
    
    history=model.fit(X_train, Y_train, nb_epoch=Epocs, batch_size=BatchSize,validation_split=ValidationSplit,verbose=0)#, callbacks=[early_stopping,checkPoint],verbose=1)
    #history=model.fit(X_train, Y_train, nb_epoch=Epocs, batch_size=BatchSize,validation_data=(X_test,Y_test), callbacks=[early_stopping,checkPoint],verbose=1)
   
    prediction= (model.predict(X_test, verbose=0))
    
    #print "################# CONFUSION MATRIX SU BEST TRAINIG #################"
    confusion_matrix(prediction,Y_test)
    
    prediction= (model.predict(X_train, verbose=0))
    confusion_matrix(prediction,Y_train)
    
    
    #outTable=str(NomeRun)+" "+str(TipoFile)+" "+str(balance)+" "+str(TypeNormalization)+" "+str(ValidationSplit)+" "+str(TypeNNet)+" "+LossFunctionString+" "+str(Optimization)+" "+str(BatchSize)+" "+str(TP)+" "+str(TN)+" "+str(FP)+" "+str(FN)+" "+str(F1)+" "+str(PREC)+" "+str(REC)+" "+str(ACC)+" "+str(seed_random)+" "+str(Patience)+"\n"
    #TableFile=open("RunTable.csv",'a')
    #TableFile.write(outTable)
    #TableFile.close()
# for b1 in range(11):
#     for b2 in range(b1+1,11):
#         print "b1,b2:",b1,b2
#         go([b1,b2])
        
Xtest,Ytest,Xtrain,Ytrain=creaDataSet([4,3])




go(None,Xtest,Ytest,Xtrain,Ytrain)