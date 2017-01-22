'''
Created on 03/ott/2016

@author: fabrizio
'''

import Parameters,sys   
from DataSet import *
from NeuralNetwork import *
from Performance import *
#import sys






DS=DataSet()

# for n in [3,4,5,6,7,8,9,11,12,13,14,15,16,17]:
#     DS.sample_train(n)
#     DS.sample_test(n)


NN=NeuralNetwork(DS)
NN.printNN()

NN.learning()

Perf=NN.testing()

Perf.write()



#stampa (Performance)


#model=Learning(P,model)

#Performace=Testing(P,model,test)


#Save(P,Performance)