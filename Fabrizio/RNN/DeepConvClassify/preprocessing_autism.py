import numpy
import os
import pdb
from skimage.draw import circle
from skimage.draw import circle_perimeter

import scipy.misc

def load_autism_data(files, input_size, test_set_perc, balanced=[True, True, True], datapath='', separator = ' ', normalize_x = False, normalize_y = False):
    
    data_nattributes=input_size[0] * input_size[1]
	
    flatten_inputsize = input_size[0] * input_size[1]
    
	# Set seed
    numpy.random.seed(23455) 

    # Set files
    datafiles = []
    print('\nLoading Autism Dataset from the files...')
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

    dataset_max = 0
	
    # Clean, convert and share datasets

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

    numpy.random.shuffle(data)																							# Shuffles train set	

    dataset_rows = data[:, 0:1065]

    # Pad data_x to flatten_inputsize
    dataset_rows = numpy.hstack((dataset_rows, numpy.zeros((dataset_rows.shape[0], flatten_inputsize-dataset_rows.shape[1]), dtype = dataset_rows.dtype))) # adds for example 1024-num cols columns to original dataset in order to have 1024 (32x32) inputs

    dataset_target = data[:, 1065:1086]
 
    print ('\nData Parameters: ')
    print ('\nAutism Dataset: ')
    print ('Dataset size : %i x %i' % (dataset_rows.shape[0], dataset_rows.shape[1]))
    print ('Target size : %i x %i' % (dataset_target.shape[0], dataset_target.shape[1]))
    print ('Sum of all Dataset values : %i' % (data.sum()));
    print ('Sum of Dataset values without target (X): %i' % (dataset_rows.sum()));
    print ('Sum of Dataset Target (Y): %i' % (dataset_target.sum()));
    
#    print ('\nBalanced training: %r' % balanced[0])
#    print ('Balanced validation set: %r' % balanced[1])
#    print ('Balanced test set: %r' % balanced[2])

 	# Last file is used to compute test set
    test_set_size = int(dataset_rows.shape[0] * test_set_perc / 100)							     		# Computes the actual size of test set
	
    # Extracts a percentage of test set from the last files 
    test_set_x = dataset_rows[0:test_set_size,:]									    					# Extracts the Test Set Rows
    test_set_y = dataset_target[0:test_set_size,:]															# Extracts the Test Set Target

    # Extracts a percentage of train set from the last files 
    train_set_x = dataset_rows[test_set_size+1:,:]									    					# Extracts the Train Set Rows
    train_set_y = dataset_target[test_set_size+1:,:]														# Extracts the Train Set Target
	
    print ('\nTraining set values size (X): %i x %i' % (train_set_x.shape[0], train_set_x.shape[1]))
    print ('Training set target vector size (Y): %i x %i' % (train_set_y.shape[0], train_set_y.shape[1]))
    print ('Sum of train set values (X): %i' % train_set_x.sum());
    print ('Sum of train set target (Y): %i' % train_set_y.sum());

    print ('\nTest set values size (X): %i x %i' % (test_set_x.shape[0], test_set_x.shape[1]))
    print ('Test set target vector size (Y): %i x %i' % (test_set_y.shape[0], train_set_y.shape[1]))
    print ('Sum of test set values (X): %i' % test_set_x.sum());
    print ('Sum of test set target (Y): %i' % test_set_y.sum());

    # Normalize the input between 0 - 1
    if normalize_x == True:
        train_set_x = train_set_x / train_set_x.max()								
        test_set_x = test_set_x / test_set_x.max()								
        
    if normalize_y == True:
        train_set_y = train_set_y / train_set_y.max()
        test_set_y = test_set_y / test_set_y.max()
		
    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    
    return rval