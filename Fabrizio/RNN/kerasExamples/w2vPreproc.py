'''
Created on 04/gen/2017

@author: fabrizio
'''


from gensim.models import word2vec
import logging,sys,numpy,pickle
#model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)


if __name__ == '__main__':


    outfile=open("word2vecDict1.txt","w")

    model= word2vec.Word2Vec.load_word2vec_format('/home/fabrizio/ECLIPSE_PYTHON/RNN/word2vect_stuff/Contenuti1.bin', binary=True)
    
    wordW2V= model.vocab.keys()
    
    


    for word in wordW2V:
        print word.encode('utf8')
        try:
                vec_word= model[word]
        except:
                "########### parola non trovata ###########"
        
        #print vec_word
        
        #".".join(vec_word)
        row=word
        for n in vec_word:
            row= row+" "+str(n)
        
        outfile.write(row.encode('utf8')+"\n")
        
    
print "End Program"
