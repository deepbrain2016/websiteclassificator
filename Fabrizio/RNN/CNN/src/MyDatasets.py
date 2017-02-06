import os,sys
import numpy
from sklearn.utils import shuffle
from keras.utils import np_utils
import pickle

class MyDatasets:
    
    
    



    def __init__(self,datasetName,NetInDim,NClasses,inputDIM,balance,norm,valid_fract,seed):
        numpy.random.seed(seed)
               
        def SameShape(train_dataset,test_dataset):
            print "SameShape: Se il Train e il Test SET non hanno la stessa larghezza inserisce degli zeri"
            W_train=train_dataset.shape[1]
            W_test=test_dataset.shape[1]
            if (W_test==W_train):
                print "... Stessa larghezza"
                return train_dataset,test_dataset
            if(W_train>W_test):
                    print "Reshape testing set"
                    n_row=test_dataset.shape[0]
                    X= numpy.zeros([n_row,W_train])
                    X[:,:W_test]=test_dataset
                    return train_dataset,X
            if(W_test>W_train):
                    print "Reshape train set"
                    print "train_dataset.shape",train_dataset.shape
                    print "test_dataset.shape",test_dataset.shape
                    n_row=train_dataset.shape[0]
                    X= numpy.zeros([n_row,W_test])
                    X[:,:W_train]=train_dataset
                    return X,test_dataset
                
        def Balance(train,test,balance):
            print "Balance: possibili bilanciamenti sono:"
            print " [ 50 (solo training 50% di 1) , 50_50  (training e testing 50% di 1), 100  (nessun bilanciamento) ]" #TODO: data la percentuale di 1 nel training costruisce un nuovo dataset"
            
            if (balance=="100") : 
                return (train,test)
            if (balance=="50" or balance=="50_50") : 
                
                    
                Q_T=len(test) # Q_T numero di record del testing di giulio
                print "Q_T: ",Q_T
                print "Unisci Train e test"
                all=numpy.append(train, test, axis=0)
                print "Separa gli uno e gli zero"
                zero= all[all[:,0]==0,:]
                print "mischio gli zero"
                zero=shuffle(zero)
                uno= all[all[:,0]==1,:]
                print "mischio gli uno"
                uno=shuffle(uno)
                N_1=len(uno)
                print "N_1: ",N_1
                
                if (balance=="50") : 
                    N_1_T=int(float(2./100*Q_T))
                    N_0_T=int(float(8./100*Q_T))
                    N_1_L=int(N_1-N_1_T)
                    N_0_L=N_1_L
                if (balance=="50_50") : 
                    N_1_T=int(float(5./100*Q_T))
                    N_0_T=int(float(5./100*Q_T))
                    N_1_L=int(N_1-N_1_T)
                    N_0_L=N_1_L                
                
                print "N_0_L: ",N_0_L
                print "N_1_L: ",N_1_L
                print "N_0_T: ",N_0_T
                print "N_1_T: ",N_1_T
                
                zero_train=zero[:N_0_L,:]
                uno_train=uno[:N_1_L,:]
                train=numpy.append(zero_train,uno_train,axis=0)
                train=shuffle(train)

                zero_test=zero[N_0_L:N_0_L+N_0_T,:]
                uno_test=uno[N_1_L:N_1_L+N_1_T,:]
                test=numpy.append(zero_test,uno_test,axis=0)
                test=shuffle(test)

                
                print "train len: ",len(train)
                print "test len: ",len(test)
                
                
            
                return (train,test)
            print "bilanciamento non valido"
            
            #sys.exit(0)
            
            return (train,test)
        
        def Shuffle(train,test):
            print "Shuffle: mescola il dataset di training e di testing"
            train=shuffle( train )
            test =shuffle( test )
            return (train,test)
        
        def Classify(train,test,NClasses):
            print "Classify: Se l'outout della rete deve essere del tipo (0,1) o (1,0) effetua una classificazione"
            if (NClasses>1):
                train = np_utils.to_categorical(train,NClasses)
                test = np_utils.to_categorical(test,NClasses)
            return (train,test)
        
        def input1Dto2D(train,test,NDim,NetInDim):
            print "input1Dto2D: Trasforma i dati di input in immagini a 2DIM in toni di grigio"
            if (NDim>1):
                def transform(Xin):
                    n_row=Xin.shape[0]
                    LarghezzaDataset=Xin.shape[1]
                    X= numpy.zeros([n_row,1,NetInDim**2])
                    X[:,0,:LarghezzaDataset]=Xin
                    X=X.reshape([n_row,1,NetInDim,NetInDim])
                    return X
                
                train = transform(train)
                test = transform(test)
                
            return (train,test)

        def Normalization(datasetName,norm,train,test):
            
            
            
            
            print "I tipi di Normalizzazione sono :('global','row','nonorm')"
            
            
            Y_train=numpy.asarray(train[:,0])
            X_train=numpy.asarray(train[:,1:])
            Y_test=numpy.asarray(test[:,0])
            X_test=numpy.asarray(test[:,1:])
        
            def nonorm(Y_train,X_train,Y_test,X_test):
                return Y_train,X_train,Y_test,X_test
            
            print "Normalization : Normalizza VAR=1 o globalmente o per riga"
            def globale(Y_train,X_train,Y_test,X_test):
                print ("Normalizzazione Globale")


                maxTrain=float(numpy.max(X_train))
                maxTest=float(numpy.max(X_test))
                
                print ("Global Max Train: ",maxTrain)
                print ("Global Max Test: ",maxTest)
                #X_train=(X_train-(maxTrain/2))/maxTrain/2
                #X_test=(X_test-(maxTrain/2))/maxTrain/2
                X_train=(X_train)/maxTrain
                X_test=(X_test)/maxTrain
                return Y_train,X_train,Y_test,X_test



                
            def row(Y_train,X_train,Y_test,X_test):
                print ("Normalizzazione per riga")
                
                maxTrainRow=X_train.max(axis=1)[:,None]
                maxTestRow=X_test.max(axis=1)[:,None]
                
                ## if ((maxTrainRow[maxTrainRow[:, 0] == 0])!=[]):
                     ## print "ci sono righe nel training set con tutti zero"
                     ## sys.exit(0)
                

                ## if ((maxTestRow[maxTestRow[:, 0] == 0])!=[]):
                     ## print "ci sono righe nel testing set con tutti zero"
                     ## sys.exit(0)

                
                X_train=(X_train)/(maxTrainRow)
                X_test=(X_test)/(maxTestRow)
                #X_train=(X_train-(maxTrainRow/2))/(maxTrainRow/2)
                #X_test=(X_test-(maxTestRow/2))/(maxTestRow/2)                
                
                
                return Y_train,X_train,Y_test,X_test
        
            
            options = {'global' : globale,'row' : row,'nonorm' : nonorm}
            
            return options[norm](Y_train,X_train,Y_test,X_test)

        self.skip=int(1)  ########### riga di zero frequenze
        #self.skip=int(4000)  ########### diverso da uno solo per debug
        self.Home="../"
        self.RunHome=self.Home+os.sep+"Run"
        self.dataHome=self.Home+os.sep+"DataSets"
        self.TrainBinaryFile="train.csv"
        self.TestBinaryFile="test.csv"
        self.TrainDebugFile="trainnew.csv"
        self.TestDebugFile="testnew.csv"
        self.TrainFreqFile="train_freq.csv"
        self.TestFreqFile="test_freq.csv"
        self.TrainTFTDFFile="train_TfIdf.csv"
        self.TestTFTDFFile="test_TfIdf.csv"
        self.Train3CHFile=""
        self.Test3CHFile=""
        
        self.TrainRECTFIDFFile="train_rec_TfIdf.csv"
        self.TestRECTFIDFFile="test_rec_TfIdf.csv"  
        
        self.TrainRECfreqFile="train_rec_freq.csv"
        self.TestTRECfreqFile="test_rec_freq.csv" 
              
        self.VisualScrPKL="trainTestVisualScrBGBIG.pkl"
        
        self.datasetName=str(datasetName)
        self.balance=balance
        self.norm=str(norm)
        self.valid_fract=float(valid_fract)

        
        self.print_param()
        
        
        if (self.datasetName=='VisualScr'):
            
            a1,a2,a3,a4=self.DispatchDatasetsPickle(self.datasetName)
            self.Ytrain=numpy.array(a1)
            self.Xtrain=numpy.array(a2)
            self.Ytest=numpy.array(a3)
            self.Xtest=numpy.array(a4)
            self.Ytrain,self.Ytest=Classify(self.Ytrain,self.Ytest,2)
            return 
        else:
        
            train_dataset,test_dataset=self.DispatchDatasets(self.datasetName)
            
            train_dataset,test_dataset=SameShape(train_dataset,test_dataset)
            
            train_dataset,test_dataset=Balance(train_dataset,test_dataset,self.balance)
            
            train_dataset,test_dataset=Shuffle(train_dataset,test_dataset)
            
            self.Ytrain,self.Xtrain,self.Ytest,self.Xtest=Normalization(self.datasetName,self.norm,train_dataset,test_dataset)
            
            self.Ytrain,self.Ytest=Classify(self.Ytrain,self.Ytest,NClasses)
            
            self.Xtrain,self.Xtest=input1Dto2D(self.Xtrain,self.Xtest,inputDIM,NetInDim)
            return
        

        
    def get(self):
        print "MyDatasets.get() ... restituisce il dataset preprocessato"
        return self.Ytrain,self.Xtrain,self.Ytest,self.Xtest
    
    
    def writeFiles(self,runName):
        
        def write_tabella_csv(fileName,table):
            #print "Se lo shape e' a tre dimensioni  "
            pickle.dump(table, fileName)
            #numpy.savetxt(fileName, table, delimiter=' ',fmt='%1.4f')  
            
        print "Crea se non esiste una cartella Run/runName"
        RunNameHome = self.RunHome+os.sep+runName
        filename = RunNameHome+os.sep+"filename"
        dir = os.path.dirname(filename)
        try:
            os.stat(dir)
        except:
            os.mkdir(dir) 
        XtrainFileName = RunNameHome+os.sep+"Xtrain.csv"
        YtrainFileName = RunNameHome+os.sep+"Ytrain.csv"
        XtestFileName = RunNameHome+os.sep+"Xtest.csv"
        YtestFileName = RunNameHome+os.sep+"Ytest.csv"
        XvalidFileName = RunNameHome+os.sep+"Xvalid.csv"
        YvalidFileName = RunNameHome+os.sep+"Yvalid.csv"
        
        print "aperto file : ",XtrainFileName
        print "aperto file : ",YtrainFileName
        print "aperto file : ",XtestFileName
        print "aperto file : ",YtestFileName
        print "aperto file : ",XvalidFileName
        print "aperto file : ",YvalidFileName
        
        XtrainWriter=open(XtrainFileName,'w')
        YtrainWriter=open(YtrainFileName,'w')
        
        XvalidWriter=open(XvalidFileName,'w')
        YvalidWriter=open(YvalidFileName,'w')

        XtestWriter=open(XtestFileName,'w')
        YtestWriter=open(YtestFileName,'w')
        
        print "Splitting training set in trainig + validation set and writes files..."
        
        row_split=int(self.Xtrain.shape[0]*(1-self.valid_fract))
        
        write_tabella_csv(YtrainWriter,self.Ytrain[:row_split])
        write_tabella_csv(XtrainWriter,self.Xtrain[:row_split,:])

        write_tabella_csv(YvalidWriter,self.Ytrain[row_split:])
        write_tabella_csv(XvalidWriter,self.Xtrain[row_split:,:])

        
        write_tabella_csv(YtestWriter,self.Ytest)
        write_tabella_csv(XtestWriter,self.Xtest)
        
        print "fine scrittura dei dataset preprocessati nei file."
        
    def print_param(self):
        print "print parametri preprocessamento DataSet"
        print  "datasetName= ", self.datasetName
        print  "balance= ",       self.balance
        print "norm= ",self.norm
        print "valid_fract= ",self.valid_fract
        
    
    def DispatchDatasets(self,datasetname):

        print "I tipi di DataSets sono :('binary','freq','TFIDF','3CH','RECfreq','RECTFIDF','debug','VisualScr')"
        
        def debug():
            print ("Debug Dataset Selected")
            return self.TrainDebugFile,self.TestDebugFile

    
        def binary():
            print ("Binary Dataset Selected")
            return self.TrainBinaryFile,self.TestBinaryFile
            
        def freq():
            print ("freq Dataset Selected")
            return self.TrainFreqFile,self.TestFreqFile
    
        def TFIDF():
            print ("TFIDF Dataset Selected")
            return self.TrainTFTDFFile,self.TestTFTDFFile
        
        def CHANNELS3():
            print ("CHANNELS3 Dataset Selected")
            return self.Train3CHFile,self.Test3CHFile

        def RECTFIDF():
            print ("RECTFIDF Dataset Selected")
            return self.TrainRECTFIDFFile,self.TestRECTFIDFFile

        def RECfreq():
            print ("RECfreq Dataset Selected")
            return self.TrainRECfreqFile,self.TestRECfreqFile

        
        options = {'binary' : binary,
                'freq' : freq,
                'TFIDF' : TFIDF,
                '3CH' : CHANNELS3,
                'RECfreq' : RECfreq,
                'RECTFIDF' : RECTFIDF,
                'debug' : debug
                
        }
        
        TrainFileName,TestFileName =options[datasetname]()

        fileTrain=self.dataHome+os.sep+TrainFileName
        
        fileTest=self.dataHome+os.sep+TestFileName
        
        TrainDataset = numpy.loadtxt(fileTrain, delimiter=" ",skiprows=self.skip)
        
        TestDataset = numpy.loadtxt(fileTest, delimiter=" ",skiprows=self.skip)
        
        return TrainDataset,TestDataset
        
    def DispatchDatasetsPickle(self,datasetname):
        if datasetname=='VisualScr':
            nomefile=self.dataHome+os.sep+self.VisualScrPKL
            return pickle.load(open(nomefile, 'rb'))
        print "just Visual Scraping DataSet"
        sys.exit(0)
   
def DebugMain():       

    DATA=MyDatasets(   'binary'     ,32      ,2       ,2       ,'100'   ,'nonorm',0.1,2345)
    Ytrain,Xtrain,Ytest,Xtest=DATA.get()


    def disegna_cerchio(X):
        #print "X.shape",X.shape
        radius=(numpy.random.randint(11-4, size=1)+4)[0] # da 7 a 10
        print "radius: ",radius
        Cx=(numpy.random.randint(15-radius, size=1))[0]+radius
        Cy=(numpy.random.randint(15-radius, size=1))[0]+radius
        coord_center=[Cx,Cy]
        print "center",coord_center
        for ix in range (radius*10):
           #print "ix",ix
                   
           iy=int((radius**2-(float(ix)/(10.))**2)**.5)
           #print "iy",iy
           px=Cx+ix/(10-1)
           py=Cy+iy
           X[0][px][py]=1
           #print px,py
           px=Cx+ix/(10-1)
           py=Cy-iy
           X[0][px][py]=1
           #print px,py
           px=Cx-ix/(10-1)
           py=Cy+iy
           X[0][px][py]=1
           #print px,py
           px=Cx-ix/(10-1)
           py=Cy-iy
           X[0][px][py]=1
           #print px,py

        #sys.exit(0)
        return X

    for i in range(Ytrain.shape[0]):
        if Ytrain[i][1]==1:
            Xtrain[i]=disegna_cerchio(Xtrain[i])

    print "Xtrain.shape",Xtrain.shape    
    Xtrain=Xtrain.reshape([Xtrain.shape[0],32*32])
    print "Xtrain.shape",Xtrain.shape    
    print "Ytrain.shape",Ytrain.shape    
    Xfile=numpy.insert(Xtrain,0,Ytrain[:,1],axis=1)
    print Xfile.shape
    Xfile=numpy.delete(Xfile,numpy.s_[1001:],axis=1)
    print Xfile.shape


    numpy.savetxt("trainnew.txt", Xfile, delimiter=' ',fmt='%i') 




    for i in range(Ytest.shape[0]):
        if Ytest[i][1]==1:
            Xtest[i]=disegna_cerchio(Xtest[i])

    print "Xtest.shape",Xtest.shape    
    Xtest=Xtest.reshape([Xtest.shape[0],32*32])
    print "Xtest.shape",Xtest.shape    
    print "Ytest.shape",Ytest.shape    
    Xfile=numpy.insert(Xtest,0,Ytest[:,1],axis=1)
    print Xfile.shape
    Xfile=numpy.delete(Xfile,numpy.s_[1001:],axis=1)
    print Xfile.shape


    numpy.savetxt("testnew.txt", Xfile, delimiter=' ',fmt='%i') 