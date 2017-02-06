list_net=['lenet500']
patience=['3']
#opt=['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
opt=['RMSprop']

ds=['binary']
bilance=['50']

#"giovagniorio"
#"cittadini"



loss=['mse','mae','mape','msle','squared_hinge','hinge','categorical_crossentropy','sparse_categorical_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity','f1']
valid=['0.3']
minibatch=['30']
i=0

for D in ds:
	for B in bilance:
		for V in valid :
			for N in list_net:
				for L in loss:
					for O in opt:
						for M in minibatch:
							for P in patience:
								print "python MainProc.py run_"+str(i)+" "+D+" "+B+" global "+V+" "+N+" "+L+" "+O+" "+M+" "+P
								i=i+1


#python MainProc.py run2_31 binary 100 global 0.1 mlp msle SGD 10
#python MainProc.py run2_32 binary 100 global 0.1 mlp msle RMSprop 10
#python MainProc.py run2_33 binary 100 global 0.1 mlp msle Adagrad 10
#python MainProc.py run2_34 binary 100 global 0.1 mlp msle Adadelta 10
#python MainProc.py run2_35 binary 100 global 0.1 mlp msle Adam 10
#python MainProc.py run2_36 binary 100 global 0.1 mlp msle Adamax 10
#python MainProc.py run2_37 binary 100 global 0.1 mlp msle Nadam 10
