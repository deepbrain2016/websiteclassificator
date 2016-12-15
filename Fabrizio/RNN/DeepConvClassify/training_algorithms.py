import numpy as np
import theano
import theano.tensor as T

# Define the last layer which is softmax
class TrainingAlgs(object):

#    def __init__(self):

    def sgd(loss, all_params, learning_rate=0.001):
        grads = T.grad(loss, all_params)                                             		# compute the gradients from errors, derivative of loss with respect to the parameters (weights)    
        updates = [                                                                         # this is the back-propagation, updates all the parameters (weights and biases) by back-propagating gradients
            (param_i, param_i - learning_rate * grad_i)                                 	# times a learning_rate factor
            for param_i, grad_i in zip(all_params, grads)
        ]
        
        return updates		

    def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,gamma=1-1e-8):
        updates = []
        all_grads = theano.grad(loss, all_params)
        alpha = learning_rate
        t = theano.shared(np.float32(1))
        b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

        for theta_previous, g in zip(all_params, all_grads):
            m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,dtype=theano.config.floatX))
            v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,dtype=theano.config.floatX))

            m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
            v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
            m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
            v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
            theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

            updates.append((m_previous, m))
            updates.append((v_previous, v))
            updates.append((theta_previous, theta) )
        
        updates.append((t, t + 1.))

        return updates