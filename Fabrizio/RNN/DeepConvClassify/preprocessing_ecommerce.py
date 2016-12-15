import numpy
import os
import pdb
from skimage.draw import circle
from skimage.draw import circle_perimeter

import scipy.misc

def load_ecommerce_data(files, input_size, valid_set_perc, test_set_perc, merge_input=True, balanced=[True, True, True], flatten_inputsize = 1024, datapath='', separator = ' ', encode_circles = True, normalize_x = False, normalize_y = False):
    
    data_nattributes=input_size[0] * input_size[1]
	
    def add_circle(dataset_x):												   # draw circles for each dataset	
        square_dataset_x=numpy.reshape(dataset_x,(input_size[0], input_size[1]))
        radius = numpy.random.randint(1, input_size[0]/2)
        xx = numpy.random.randint(radius, input_size[0]-radius)
        yy = numpy.random.randint(radius, input_size[1]-radius)
	
        #rr, cc = circle_perimeter(xx, yy, radius)						       # draw a circle onto the image
        rr, cc = circle(xx, yy, radius)									       # draw a circle onto the image
        square_dataset_x[rr, cc] = 1
        
        return numpy.reshape(square_dataset_x,(1, data_nattributes))
	
    def fill_circles(dataset_x,radius):    										# convert pixels into circles	
        square_dataset_x=numpy.reshape(dataset_x,(input_size[0], input_size[1]))
        new_image_from_dataset=numpy.zeros((input_size[0] * 2 * radius, input_size[1] * 2 * radius))
        for y in range (0, input_size[1]-1):
            for x in range (0, input_size[0]-1):
                if square_dataset_x[x,y] > 0.0:								    # inserts only the circles where there is a value			
                    xx = x * 2 * radius + radius
                    yy = y * 2 * radius + radius
                    #rr, cc = circle_perimeter(xx, yy, radius)						# draw a circle onto the image
                    rr, cc = circle(xx, yy, radius)		  					        # draw a circle onto the image
                    new_image_from_dataset[rr, cc] = 1

        return numpy.reshape(new_image_from_dataset,(1, (input_size[0] * 2 * radius) * (input_size[1] * 2 * radius)))
    
	# Set seed
    numpy.random.seed(23455) 

    # Set files
    datafiles = []
    print('\nLoading Istat Datasets from the files...')
    num_files = numpy.asarray(files).shape[0]
    for f in range(num_files): 
        datafiles.append(files[f])
        print ('Original training set file : %s' % files[f])

    print ('Original test set: %s' % files[num_files-1])

    # Read from file datasets: train sets and test set
    for i in range(0,num_files):
        data_dir, data_file = os.path.split(datafiles[i])

        if data_dir == "" and not os.path.isfile(datafiles[i]):
            # Check if dataset is in the data directory.
            new_path = os.path.join(                   			                 # compose the path where training, validation and test sets are
                os.path.split(__file__)[0],
				datapath, 	
                datafiles[i]
            )
        
        if os.path.isfile(new_path) or data_file == datafiles[i]:
            datafiles[i] = new_path

    datasets_positives = [[],[]]
    datasets_negatives = [[],[]]
    garbage = []
    dataset_max = 0
	
    # Clean, convert and share datasets
    # Balance positives and negatives 

    for k in range(num_files):
        data = []
        
        print ('Reading: %s' % datafiles[k]) 
        file = open(datafiles[k], 'r')
        i = 0
        for line in file:
            splittedline = line.split(separator)                       				                    # with the first column - target
            data.append(splittedline)
            i+=1 
    
        print ('%i lines read' % i)
        print ('Number of attributes: %i' % numpy.asarray(data).shape[1])
        data = numpy.delete(data, (0), axis = 0).astype(numpy.float)                                    # delete the head row and converts to float
        
        # Pad data_x to flatten_inputsize
        data = numpy.hstack((data, numpy.zeros((data.shape[0], flatten_inputsize-data.shape[1]+1), dtype = data.dtype))) # adds for example 1024-num cols columns to original dataset in order to have 1024 (32x32) inputs

        # If merge = true puts all the data in the first rows of datasets [0]
        if merge_input == True:
	        i=0		
        else:
            i=k       	

        # Split positives and negatives
        for j in range(0, data.shape[0]):
            if data[j,0] == 1.0:																# positive target			
                datasets_positives[i].append(data[j,:])
            elif data[j,0] == 0.0:																# negative target
                datasets_negatives[i].append(data[j,:])
            else: 
                print (data[j,:])
                garbage.append(data[j,:])

	
	# Converts positives and negatives from list to arrays
    for k in range(num_files):
        datasets_positives[k] = numpy.asarray(datasets_positives[k])
        datasets_negatives[k] = numpy.asarray(datasets_negatives[k])
        garbage = numpy.asarray(garbage)

	# If merge = true puts the max number of positive and negative datasets pairs is 1 otherwise 2 (or n) 
    if merge_input == True:
        idx_datasets_pos_neg = 1		
    else:
        idx_datasets_pos_neg = num_files-1		

    print ('\nData Parameters: ')

    dataset_rows = 0
    dataset_sum = 0
    dataset_sum_without_target = 0
    dataset_sum_target = 0
    
    end_point = idx_datasets_pos_neg
    if merge_input == False: 
        end_point = idx_datasets_pos_neg + 1
	
    for k in range(end_point):
        dataset_rows += datasets_positives[k].shape[0] + datasets_negatives[k].shape[0]
        dataset_sum += datasets_positives[k].sum() + datasets_negatives[k].sum()
        dataset_sum_without_target += datasets_positives[k][:,1:].sum()+datasets_negatives[k][:,1:].sum()
        dataset_sum_target += datasets_positives[k][:,0].sum()+datasets_negatives[k][:,0].sum()   
    
    print ('\nECommerce Dataset: ')
    print ('Dataset size (without 2 head rows): %i x %i' % (dataset_rows, datasets_positives[0].shape[1]))
    print ('Sum of all Dataset values : %i' % (dataset_sum));
    print ('Sum of Dataset values without target (X): %i' % (dataset_sum_without_target));
    print ('Sum of Dataset Target (Y): %i' % (dataset_sum_target));
    print ('Garbage size: %i x %i' % (garbage.shape[0], garbage.shape[0]))
    
    print ('\nMerged and enlarged training set: %r' % merge_input)
    print ('\nBalanced training: %r' % balanced[0])
    print ('Balanced validation set: %r' % balanced[1])
    print ('Balanced test set: %r' % balanced[2])

    valid_set_positives_size = 0
    valid_set_negatives_size = 0
    test_set_positives_size = 0
    test_set_negatives_size = 0
	
	# First files are used to compute train and validation set 
    for k in range(idx_datasets_pos_neg):
        valid_set_positives_size += int(datasets_positives[k].shape[0] * valid_set_perc / 100)												# Computes the actual size of validation set
        valid_set_negatives_size += int(datasets_negatives[k].shape[0] * valid_set_perc / 100)							     				# Computes the actual size of validation set
	
    test_point = idx_datasets_pos_neg
    if merge_input == True: 
        test_point = 0
	# Last file is used to compute test set
    test_set_positives_size = int(datasets_positives[test_point].shape[0] * test_set_perc / 100)							     		# Computes the actual size of test set
    test_set_negatives_size = int(datasets_negatives[test_point].shape[0] * test_set_perc / 100)					    				# Computes the actual size of test set


    for k in range(idx_datasets_pos_neg):
        # Extracts a percentage of positives from the first files 
        valid_set_positives_k_size = int(datasets_positives[k].shape[0] * valid_set_perc / 100)
        if k == 0: 
            valid_set_positives = datasets_positives[k][0:valid_set_positives_k_size,:]  						    						 	# Extracts the Validation Set Positives
        else: 
            valid_set_positives = numpy.vstack((valid_set_positives, datasets_positives[k][0:valid_set_positives_k_size,:]))								    						 	# Extracts the Validation Set Positives
        datasets_positives[k] = numpy.delete(datasets_positives[k], numpy.arange(0,valid_set_positives_k_size), axis=0).astype(numpy.float)		# Deletes the extracted rows
        # Extracts a percentage of negatives from the training file or last file 
        valid_set_negatives_k_size = int(datasets_negatives[k].shape[0] * valid_set_perc / 100)
        if k == 0: 
            valid_set_negatives = datasets_negatives[k][0:valid_set_negatives_k_size,:]		     											# Extracts the Validation Set Positives
        else:    
            valid_set_negatives = numpy.vstack((valid_set_negatives, datasets_negatives[k][0:valid_set_negatives_k_size,:]))				     			     											# Extracts the Validation Set Positives
        datasets_negatives[k] = numpy.delete(datasets_negatives[k], numpy.arange(0,valid_set_negatives_k_size), axis=0).astype(numpy.float)		# Deletes the extracted rows
	

	
    # Extracts a percentage of positives from the last files 
    test_set_positives = datasets_positives[test_point][0:test_set_positives_size,:]															    					# Extracts the Test Set Positives
    datasets_positives[test_point] = numpy.delete(datasets_positives[test_point], numpy.arange(0,test_set_positives_size), axis=0).astype(numpy.float)		# Deletes the extracted rows
    test_set_negatives = datasets_negatives[test_point][0:test_set_negatives_size,:]														    						# Extracts the Test Set Positives
    datasets_negatives[test_point] = numpy.delete(datasets_negatives[test_point], numpy.arange(0,test_set_negatives_size), axis=0).astype(numpy.float)	   	# Deletes the extracted rows

    # If unbalanced, datasets preserve the original unbalancing
	# The rest of Datasets goes into the train set
    # If balanced[0] = true, then training set must be balanced, then removes the exceeding negatives or positives, namely there is a balance between positives and negatives 
    for k in range(idx_datasets_pos_neg):
        if balanced[0] == True:
            if datasets_positives[k].shape[0] > datasets_negatives[k].shape[0]:										# Truncate positives
                datasets_positives[k] = numpy.delete(datasets_positives[k], numpy.arange(0,datasets_positives[k].shape[0]-datasets_negatives[k].shape[0]), axis=0).astype(numpy.float)
            elif datasets_positives[k].shape[0] < datasets_negatives[k].shape[0]:									# Truncate Negatives
                datasets_negatives[k] = numpy.delete(datasets_negatives[k], numpy.arange(0,datasets_negatives[k].shape[0]-datasets_positives[k].shape[0]), axis=0).astype(numpy.float)
    
        if k == 0: 
            train_set = numpy.vstack((datasets_positives[k], datasets_negatives[k]))											# Merges positives and negatives
        else:
            train_set = numpy.vstack((train_set, numpy.vstack((datasets_positives[k], datasets_negatives[k]))))		  		    # Merges positives and negatives
		
    numpy.random.shuffle(train_set)																								# Shuffles train set	
    train_set_y = train_set[:,0] 																								# extracts the target			
    train_set_x = numpy.delete(train_set, (0), axis = 1)				                            							# delete the first column (target) from the training  s
    print ('\nTraining set values size (X): %i x %i' % (train_set_x.shape[0], train_set_x.shape[1]))
    print ('Training set target vector size (Y): %i x 1' % train_set_y.shape[0])
    print ('Sum of train set values (X): %i' % train_set_x.sum());
    print ('Sum of train set target (Y): %i' % train_set_y.sum());

    # If balanced[1] = true, then validation set must be balanced, then removes the exceeding negatives or positives, namely there is a balance between positives and negatives 
    if balanced[1] == True:
        if valid_set_positives.shape[0] > valid_set_negatives.shape[0]:												# Truncate positives
            valid_set_positives = numpy.delete(valid_set_positives, numpy.arange(0,valid_set_positives.shape[0]-valid_set_negatives.shape[0]), axis=0).astype(numpy.float)
        elif valid_set_positives.shape[0] < valid_set_negatives.shape[0]:									        # Truncate Negatives
            valid_set_negatives = numpy.delete(valid_set_negatives, numpy.arange(0,valid_set_negatives.shape[0]-valid_set_positives.shape[0]), axis=0).astype(numpy.float)

    valid_set = numpy.vstack((valid_set_positives, valid_set_negatives))											# Merges positives and negatives
    numpy.random.shuffle(valid_set)																					# Shuffles the validation set	
    valid_set_y = valid_set[:,0] 													    							# extracts the target			
    valid_set_x = numpy.delete(valid_set, (0), axis = 1)					                            			# delete the first column (target) from the valid set  
    #valid_set_x = numpy.hstack((valid_set_x, numpy.zeros((valid_set_x.shape[0], 24), dtype = valid_set_x.dtype)))   # adds 24 columns to original dataset in order to have 1024 (32x32) inputs
    print ('\nValidation set values size (X): %i x %i' % (valid_set_x.shape[0], valid_set_x.shape[1]))
    print ('Validation set target vector size (Y): %i x 1' % valid_set_y.shape[0])
    print ('Sum of validation set values (X): %i' % valid_set_x.sum());
    print ('Sum of validation set target (Y): %i' % valid_set_y.sum());

    # If balanced[2] = true, then validation set must be balanced, then removes the exceeding negatives or positives, namely there is a balance between positives and negatives 
    if balanced[2] == True:
        if test_set_positives.shape[0] > valid_set_negatives.shape[0]:												# Truncate positives
            test_set_positives = numpy.delete(test_set_positives, numpy.arange(0,test_set_positives.shape[0]-test_set_negatives.shape[0]), axis=0).astype(numpy.float)
        elif test_set_positives.shape[0] < valid_set_negatives.shape[0]:									        # Truncate Negatives
            test_set_negatives = numpy.delete(test_set_negatives, numpy.arange(0,test_set_negatives.shape[0]-test_set_positives.shape[0]), axis=0).astype(numpy.float)

    test_set = numpy.vstack((test_set_positives, test_set_negatives))											# Merges positives and negatives
    numpy.random.shuffle(test_set)																				# Shuffles test set	
    test_set_y = test_set[:,0] 													    							# extracts the target			
    test_set_x = numpy.delete(test_set, (0), axis = 1)					                            			# delete the first column (target) from the test set
    #test_set_x = numpy.hstack((test_set_x, numpy.zeros((test_set_x.shape[0], 24), dtype = test_set_x.dtype)))   # adds 24 columns to original dataset in order to have 1024 (32x32) inputs
    print ('\nTest set values size (X): %i x %i' % (test_set_x.shape[0], test_set_x.shape[1]))
    print ('Test set target vector size (Y): %i x 1' % test_set_y.shape[0])
    print ('Sum of test set values (X): %i' % test_set_x.sum());
    print ('Sum of test set target (Y): %i' % test_set_y.sum());

    # Normalize the input between 0 - 1
    # Normalize the input between 0 - 1
    if normalize_x == True:
        train_set_x = train_set_x / train_set_x.max()								
        valid_set_x = valid_set_x / valid_set_x.max()								
        test_set_x = test_set_x / test_set_x.max()								
        
    if normalize_y == True:
        train_set_y = train_set_y / train_set_y.max()
        valid_set_x = valid_set_y / valid_set_y.max()								
        test_set_y = test_set_y / test_set_y.max()

    # Add cicles to the train set where target is 1
    # Split positives and negatives
    if encode_circles == True: 
        '''
        # insert a circle inside of images - trivial case - test, great classification 
		for j in range(0, train_set_y.shape[0]):
            if train_set_y[j] == 1.0:																	# positive target			
                train_set_x[j,:]=add_circle(train_set_x[j,:])
        for j in range(0, valid_set_y.shape[0]):
            if valid_set_y[j] == 1.0:																	# positive target			
                valid_set_x[j,:]=add_circle(valid_set_x[j,:])
        for j in range(0, test_set_y.shape[0]):
            if test_set_y[j] == 1.0:																	# positive target			
                test_set_x[j,:]=add_circle(test_set_x[j,:])
        '''	
        
		# New enlarged datasets
        radius = 224 / input_size[0] / 2
        new_train_set_x = numpy.zeros((train_set_x.shape[0], (int(input_size[0] * 2 * radius)) * (int(input_size[1] * 2 * radius))), dtype='float32')
        new_valid_set_x = numpy.zeros((valid_set_x.shape[0], (int(input_size[0] * 2 * radius)) * (int(input_size[1] * 2 * radius))), dtype='float32')
        new_test_set_x = numpy.zeros((test_set_x.shape[0], (int(input_size[0] * 2 * radius)) * (int(input_size[1] * 2 * radius))), dtype='float32')

        for j in range(0, train_set_y.shape[0]):
            new_train_set_x[j,:]=fill_circles(train_set_x[j,:],radius)
        for j in range(0, valid_set_y.shape[0]):
            new_valid_set_x[j,:]=fill_circles(valid_set_x[j,:],radius)
        for j in range(0, test_set_y.shape[0]):
            new_test_set_x[j,:]=fill_circles(test_set_x[j,:],radius)

        square_train_set_x=numpy.reshape(new_train_set_x[2,:],(input_size[0] * 2 * radius, input_size[1] * 2 * radius))
        scipy.misc.imsave('outfile1.jpg', square_train_set_x)

        train_set_x = new_train_set_x
        valid_set_x = new_valid_set_x
        test_set_x = new_test_set_x
	
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    
    return rval