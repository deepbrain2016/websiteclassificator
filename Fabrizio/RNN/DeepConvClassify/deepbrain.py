# Framework selection 
use_keras = True			# Use Keras Implementation: Keras has complex architectures implementations (VGG, GoogleNet, etc.)  
use_theano = False			# Use a Pure Theano Implementation. Theano has simple more in depth architectures implementations (Lenet, etc.) 	

import pickle
import pickle
import gzip
import numpy
import os
import sys
import timeit
import winsound
import math

import pdb

from preprocessing_ecommerce import load_ecommerce_data
from preprocessing_autism import load_autism_data

if use_keras == True:
	# Keras imports
    from klenet import LeNet
    from kecommercenet import Ecommercenet
    from kautismnet import AutismNet
    from kalexnet import AlexNet
    from kvgg16 import VGG_16
    from kgooglenet import GoogleNet
    from keras.optimizers import SGD
    from keras.utils import np_utils
    from keras.metrics import fbeta_score

if use_theano == True:
	# Theano imports
    import theano
    import theano.tensor as T
    from theano import shared
    from sympy.core.tests.test_wester import test_Y1
    from gpu_memory import shared_dataset
    from lenet import Lenet
    from ecommercenet import Ecommercenet

# Globals

# Domain Selection 
autism = False
ecommerce = True

if autism == True: 
    # Autism

    # Input
    trainset_file = 'autism.csv'
    testset_file = ''

    # straight datasets
    datasets_path = 'autism'

    # separators
    field_separator = ';'                                        # CSV field separator     
    test_set_perc = 3.7                                           # Test set percentage with respect to the Training Set 

    merge_input = False
    balance_train_set = True 									  # Balance Positives and Negatives within the training set
    balance_valid_set = False 									  # Balance Positives and Negatives within the validation set
    balance_test_set = False 									  # Balance Positives and Negatives within the test set

    normalize_x = False                                           # Normalize input between 0 and 1
    normalize_y = True                                           # Normalize input between 0 and 1

    encode_circles = False   									  # Encode pixels as circles, only in keras implementation 	

    # Hyperparameters

    # General settings
    epochs_number = 100
    learning_rate_float = 0.0015							      # best learning rate = 0.045 at moment, 0.001 on mcover
    number_of_batches = 1
    input_height = 33											  # Height of the input image																																																	
    input_width = 33											  # Width of the input image																																																	
    output_size = 21
	
    # Sound
    sound_freq = 1000 										      # Set Frequency in Hertz
    sound_dur = 3000 											  # Set Duration in ms, 1000 ms == 1 second

elif ecommerce == True: 
    # E-commerce

    # Input
    trainset_file = 'train.csv'
    testset_file = 'test.csv' 

    # straight datasets
    #datasets_path = 'istatdata'
    #datasets_path = 'istatdatadiegobalanced'
    datasets_path = 'ecommercebasic'
    #datasets_path = 'ecommercefreq'
    #datasets_path = 'ecommerceTfIdf'

    # recoded datasets
    #datasets_path = 'ecommercefreq_recoded'
    #datasets_path = 'ecommerceTfIdf_recoded'
    #datasets_path = 'ecommercebasic_mcover'

    # separators
    #field_separator = ','                                        # CSV field separator     
    field_separator = ' '                                         # CSV field separator     

    valid_set_perc = 3.7                                          # Validation set percentage with respect to the Training Set
    test_set_perc = 100                                           # Test set percentage with respect to the Training Set 

    merge_input = False
    balance_train_set = True 									  # Balance Positives and Negatives within the training set
    balance_valid_set = False 									  # Balance Positives and Negatives within the validation set
    balance_test_set = False 									  # Balance Positives and Negatives within the test set

    normalize_x = False                                           # Normalize input between 0 and 1
    normalize_y = False                                            # Normalize input between 0 and 1

    encode_circles = True   									  # Encode pixels as circles, only in keras implementation 	

    # Hyperparameters

    # General settings
    epochs_number = 3
    learning_rate_float = 0.045							      # best learning rate = 0.045 at moment, 0.001 on mcover
    number_of_batches = 1
    input_height = 32											  # Height of the input image																																																	
    input_width = 32											  # Width of the input image																																																	
    output_size = 2
	
    # Sound
    sound_freq = 1000 										      # Set Frequency in Hertz
    sound_dur = 3000 											  # Set Duration in ms, 1000 ms == 1 second
	
# Output Images
outputimages_path = 'images/input'

def evaluate_convnet(input_size=[input_height, input_width], 
                     learning_rate=learning_rate_float, n_epochs=epochs_number, batch_size=number_of_batches,
					 files=[trainset_file, testset_file], encode_circles = encode_circles):

    start_time = 0
    global_start_time = timeit.default_timer()

    if use_keras == True:
        print('\nUsing Keras...')	
    elif use_theano == True: 
        print('\nUsing Theano...') 	
        encode_circles = False						# Cannot use this modality in pure Theano implementation at moment
  	
	
	# spread datasets onto all the input training sets, validation sets and testing sets
    if autism == True: 
        datasets = load_autism_data(files=[trainset_file], input_size = input_size, test_set_perc=test_set_perc, balanced=[balance_train_set, balance_valid_set, balance_test_set], datapath=datasets_path, separator = field_separator, normalize_x = normalize_x, normalize_y = normalize_y)
        hostmem_train_set_x, hostmem_train_set_y = datasets[0]
        hostmem_test_set_x, hostmem_test_set_y = datasets[1]
    elif ecommerce == True: 
        datasets = load_ecommerce_data(files=[trainset_file, testset_file], input_size = input_size, valid_set_perc=valid_set_perc, test_set_perc=test_set_perc, merge_input=merge_input, balanced=[balance_train_set, balance_valid_set, balance_test_set], flatten_inputsize = input_size[0]*input_size[1], datapath=datasets_path, separator = field_separator, encode_circles = encode_circles, normalize_x = normalize_x, normalize_y = normalize_y)
        hostmem_train_set_x, hostmem_train_set_y = datasets[0]
        hostmem_valid_set_x, hostmem_valid_set_y = datasets[1]
        hostmem_test_set_x, hostmem_test_set_y = datasets[2]
 		
        mul_factor = 1
        if encode_circles == True: 
            mul_factor = 7	
            input_size = [int(math.sqrt(hostmem_train_set_x.shape[1])), int(math.sqrt(hostmem_train_set_x.shape[1]))]			# re-compute the input_size

    print('Encode input circles: %r' % encode_circles) 	
    print
    print ('\nBatch size: %i' % batch_size)
    print

    # compute number of minibatches for training, validation and testing
    n_train_batches = hostmem_train_set_x.shape[0] // batch_size
    if ecommerce == True:
	    n_valid_batches = hostmem_valid_set_x.shape[0] // batch_size
    n_test_batches = hostmem_test_set_x.shape[0] // batch_size
	
    print ('Number of training batches: %i' % n_train_batches)
    if ecommerce == True:
        print ('Number of validation batches: %i' % n_valid_batches)
    print ('Number of test batches: %i' % n_test_batches)

    print ('\nTraining of the neural network...')

    # Keras Implementation 
    if use_keras == True: 
		# Reshape and add an axis to each data set in order to pass it to the conv layer. Transform labels into integer categorical arrays too. 	
        train_set_x, train_set_y = [hostmem_train_set_x.reshape((hostmem_train_set_x.shape[0],input_size[0],input_size[1]))[:,numpy.newaxis,:,:], np_utils.to_categorical(hostmem_train_set_y.astype("int"), output_size)]
        test_set_x, test_set_y = [hostmem_test_set_x.reshape((hostmem_test_set_x.shape[0],input_size[0],input_size[1]))[:,numpy.newaxis,:,:],  np_utils.to_categorical(hostmem_test_set_y.astype("int"), output_size)]
        
        opt = SGD(lr=learning_rate_float)					# Training Algorithm

        if autism == True:
            #deepnetwork = LeNet.build(width=32, height=32, depth=1, classes=2)	
            deepnetwork = AutismNet.build(width=input_size[0], height=input_size[1], depth=1, classes=output_size)	
        elif ecommerce == True:
            #deepnetwork = Ecommercenet.build(width=input_size[0], height=input_size[1], depth=1, classes=output_size, mul_factor=mul_factor)	
            deepnetwork = AlexNet.build(width=input_size[0], height=input_size[1], depth=1, classes=2, mul_factor=mul_factor)
            #deepnetwork = VGG_16.build(width=input_size[0], height=input_size[1], depth=1, classes=2, mul_factor=mul_factor)
            #deepnetwork = GoogleNet.build(width=input_size[0], height=input_size[1], depth=1, classes=2, mul_factor=mul_factor)
        
        deepnetwork.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
        start_time = timeit.default_timer()
        deepnetwork.fit(train_set_x, train_set_y, batch_size=batch_size, nb_epoch=epochs_number, verbose=2)	
        deepnetwork.evaluate(test_set_x, test_set_y, verbose = 2)
        #pdb.set_trace()
	
	# Pure Theano Implementation
    if use_theano == True: 
	    # Conversion to SHARED Theano variables is needed in order to boost GPU computation, as the delays from the host-device transfer would erase
        # the GPU parallelism upside. Putting data in device's shared memory, multiple data-transfer are not necessary anymore.  
        #mem = GPUMemory([])
        train_set_x, train_set_y = shared_dataset([hostmem_train_set_x, hostmem_train_set_y])
        valid_set_x, valid_set_y = shared_dataset([hostmem_valid_set_x, hostmem_valid_set_y])                    # to avoid data transfer delays when using gpu
        test_set_x, test_set_y = shared_dataset([hostmem_test_set_x, hostmem_test_set_y])
        #test_set_x, test_set_y = [hostmem_test_set_x, hostmem_test_set_y]										 # test is on the host memory to avid out-of-memory on the gpu memory 	

        index = T.lscalar()             # index of minibatch
        x = T.matrix('x')               # image provided to the input of the convolutional network
        y = T.ivector('y')              # labels are int sequences

        #deepnetwork = Lenet(index, x, y, batch_size=batch_size, input_size=input_size, learning_rate=learning_rate, datasets=[[train_set_x, train_set_y], [valid_set_x, valid_set_y], [test_set_x, test_set_y]])
        deepnetwork = Ecommercenet(index, x, y, batch_size=batch_size, input_size=input_size, learning_rate=learning_rate, datasets=[[train_set_x, train_set_y], [valid_set_x, valid_set_y], [test_set_x, test_set_y]], mul_factor=mul_factor)

        # Model parameters
        patience = 10000                                                         
        patience_increase = 2
    
        improvement_threshold = 0.995
    
        validation_frequency = min(n_train_batches, patience // 2)

        #print "Validation frequency: ", validation_frequency
    
        best_validation_loss = numpy.inf
        best_iter = 0
        best_score = 0
        epoch = 0
        done_looping = False
    
        start_time = timeit.default_timer()
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
        
            for minibatch_index in range(n_train_batches):                      					# from 0 to 99 for example are the current number of batches
                iter = (epoch - 1) * n_train_batches+minibatch_index            					# select the correct mini batch for the iteration
        
                #if iter % 10 == 0:                                            						# Every multiple of 100 prints the iteration number
                #    print "Training Iteration: ", iter
        
                cost_ij = deepnetwork.train_model(minibatch_index)                             					# call the train model function
                #print "Batch index: ", minibatch_index, " - Cost: ", cost_ij
            
                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [deepnetwork.validate_model(i) for i in range(n_valid_batches)]
                
                    this_validation_loss = numpy.mean(validation_losses)

                    #print "Validation losses mean:", this_validation_loss

                    print('epoch %i, minibatch %i/%i, validation error %f %%, validation set number of batches %i' % (epoch, minibatch_index + 1, n_train_batches,
                                                                                                                           this_validation_loss * 100., n_valid_batches))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                            improvement_threshold:
                            patience = max(patience, iter * patience_increase)						# increase the patienze

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        '''
					    # test it on the test set
                        test_losses = [
                            deepnetwork.test_model(i) for i in range(n_test_batches)
                        ]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                          'best model %f %%') % (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
                    '''
                
                if patience <= iter:																# stops when patience <= iter			
                    done_looping = True
                    break
    
        print('Optimization complete.')
        #print('Best validation score of %f %% obtained at iteration %i, '
        #      'with test performance %f %%' %
        #      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print('Best validation score of %f %% obtained at iteration %i' % (best_validation_loss * 100., best_iter + 1))
    
    end_time = timeit.default_timer()

    print ('\nTraining time: %.2f minutes' % ((end_time - start_time) / 60.))
    print ('Global time: %.2f minutes\n' % ((end_time - global_start_time) / 60.))

    print ('Testing the trained network...')
    
    if use_keras == True:
        prediction_classes = deepnetwork.predict_classes(test_set_x, batch_size = batch_size, verbose = 0)
        prediction_output = numpy.vstack((prediction_classes, hostmem_test_set_y.astype("int")))
		
    if use_theano == True: 
        prediction_output_in_batches = numpy.asarray([deepnetwork.predict_model(i) for i in range(n_test_batches)]) # it must work like batches as layer1 tensor and all are sized with batchs sizes
        
        # test it on the test set
        test_errors = [deepnetwork.test_model(i) for i in range(n_test_batches)]
        test_error_score = numpy.mean(test_errors)

	    # Calculate the scores over all the outputs
        prediction_output = prediction_output_in_batches[0,:,:] 
        for i in range(1, prediction_output_in_batches.shape[0]):
            prediction_output = numpy.hstack((prediction_output, prediction_output_in_batches[i,:,:]))  
 	
    print ('\nPredicted output: ') 
    print (prediction_output[0]) 
    print ('\nReal output: ') 
    print (prediction_output[1]) 
	
    TN = 0
    FN = 0
    TP = 0
    FP = 0

    pred_y = prediction_output[0]
    print ('\nSum of Predicted values: %i' % pred_y.sum())
    print
    real_y = prediction_output[1]
    print ('Sum of Real values: %i' % real_y.sum())
    print
	
    #Computation of true negative, fakse negative, true positive, false positive outputs
    for j in range(0, pred_y.shape[0]):
        if pred_y[j] == 0 and real_y[j]==pred_y[j]:
            TN += 1
        if pred_y[j] == 0 and real_y[j]!=pred_y[j]:
            FN += 1
        if pred_y[j] == 1 and real_y[j]==pred_y[j]:
            TP += 1
        if pred_y[j] == 1 and real_y[j]!=pred_y[j]:
            FP += 1
                
    print ('Number of True Negatives %i' % TN)
    print ('Number of False Negatives %i' % FN)
    print ('Number of True Positives %i' % TP)
    print ('Number of False Positives %i' % FP)
    
    precision = TP/(TP+FP)                              # rate of true positives over all positives, number of matches which are relevant to the user's need
    recall = float(TP/(TP+FN))                          # rate of true positives over true positives and false negatives, sensitivy is the probability that a relevant match is retrieved   
    P = TP + FN											# All the positives are all the true positives + all false negatives (which are positives) 
    N = TN + FP										    # All the negatives are all the true negatives + all false positives (which are negatives) 
			
    accuracy=float((TP+TN)/(TP+TN+FP+FN))
    f1_score=float((2*TP)/(2*TP+FP+FN))                 # harmonic mean of precision and sensitivity
    TPTN_rates = float(TP/P+TN/N)						# Rate of TP/P + TN/N, which is steady according to the distribution of positives and negatives withing the input for the same model unlike f1-score and accuracy
	
    print ('Measures: ')
    print ('\nPrecision  : %f' % precision)
    print ('Recall   : %f' % recall)
    print ('Accuracy  : %f' % accuracy)
    print ('F1 score   : %f' % f1_score)
   
    if use_theano == True: 
 	    print ('Test Error Score   : %f %%' % (test_error_score * 100))
    
    print ('True positives rate + True negatives rate : %f' % TPTN_rates)
	
    # Emits a sound warning the processing is finished
    winsound.Beep(sound_freq, sound_dur)	

if __name__ == '__main__':
    evaluate_convnet()
    
#def experiment(state, channel):
#    evaluate_lenet5(state.learning_rate, dataset=state.dataset)