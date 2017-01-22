'''
Created on 04/gen/2017

@author: fabrizio
'''


import os
import shutil
import sys
import random

k_mix=50 # percentuale della classe mix che passa all'altra classe
k_no_mix=50 # percentuale della classe no_mix che passa all'altra classe

i=0
flag=0
nome_dir='/home/fabrizio/git/websiteclassificator/Fabrizio/RNN/kerasExamples/20_newsgroup'
out_dir="/home/fabrizio/git/websiteclassificator/Fabrizio/RNN/kerasExamples/2_newsgroupSbil/0mix"
out_dir_good="/home/fabrizio/git/websiteclassificator/Fabrizio/RNN/kerasExamples/2_newsgroupSbil/9EC"

n_scambio_mix=0
n_scambio_nomix=0
for root,dirs,files in os.walk(nome_dir):
    print "root",root
    print "dirs",dirs
    #print "files",files
    print flag
    if flag<2:

        for f in files:
            f1=root+os.sep+f
            f2=out_dir_good+os.sep+f+'000000'+str(i)
	    if random.randint(0,99)< k_mix:
        	f1=root+os.sep+f
        	f2=out_dir+os.sep+f+'0001000'+str(i)
   	        n_scambio_nomix+=1		
            #print f1,f2
            shutil.copy(f1,f2) 
        flag+=1
        continue
    #sys.exit(0)

    print "directories for the mixing"
    print "ciclo per i primi 201 file di ogni directory"
    for f in files[:(20*10+1)]:
       
        f1=root+os.sep+f
        f2=out_dir+os.sep+f+'000000'+str(i)
	if random.randint(0,99)< k_no_mix:
            f1=root+os.sep+f
            f2=out_dir_good+os.sep+f+'0009000'+str(i)
	    n_scambio_mix+=1		
        #print f1,f2
        shutil.copy(f1,f2 )
    i+=1    
print "Scambio no_mix!!!!! ",n_scambio_nomix
print "Scambio mix!!!!! ",n_scambio_mix
