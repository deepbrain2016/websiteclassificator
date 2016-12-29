import theano


import theano.tensor as T


# Define all types of activation functions 
class ActivationFunctions(object):

    def __init__(self, input):
        self.input = input