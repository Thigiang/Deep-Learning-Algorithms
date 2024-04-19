import numpy as np
from activation_fnc import Activation
class dnn:


    def intitialize_parameters_deep(self, layer_sizes):
        """
        input:
         -- layer_sizes: a list of array that contains the size of all the layers (including the input layer 0)
        return:
        -- parameters: a dictionary with initialized parameters for all the layers.
        """
        L = len(layer_sizes)
        parameters = {}
        for l in range(1, L):
            n_x, n_h= layer_sizes[l-1], layer_sizes[l]
            parameters["W"+str(l)]= np.random.randn(n_h, n_x)*0.01
            parameters["b"+str(l)]= np.zeros((n_h, 1))

        return parameters
    
    def compute_cost(self, AL, Y):
        """
        Input:
        -- AL: final output (output of the final layer) or predicted Y. This is matrix of size (1, m)
        -- Y: true labels - matrix of size (1, m)
        Return:
        -- cost: the cost of the algorithm

        """
        m = Y.shape[1]
        loss = np.multiply(Y, np.log(AL))+np.multiply(1-Y, np.log(1-AL))
        cost = (-1/m)*np.sum(loss)
        return cost
    

    def forward_prop(self, A_prev, W, b, activation):
        """
        Input:
        -- A_prev: activations from previous layer
        -- W: weight parameter for current layer
        -- b: bias for current layer
        Return:
        -- A: activation for current layer
        -- cache: dictionary contains Z, W, b for current layer
        """
        Z = np.dot(W, A_prev)+b
        if activation == 'sigmoid':
            A = Activation.sigmoid(Z)
        elif activation == 'tanh':
            A=Activation.tanh_fnc(Z)
        elif activation == 'relu':
            A = Activation.relu(Z)
        else:
            A = Activation.leaky_relu(Z)
        cache = {"Z": Z,
                 "W": W,
                 "b": b}
        return A, cache
    
    def L_forward_prop(self, X, parameters):
        L=len(parameters) // 2
        caches = []
        A_prev = X
        for l in range(1, L):
            A_prev, cache = self.forward_prop(A_prev, parameter["W"+str(l)], parameter["b"+str(l)], "relu")
            caches.append(cache)
        AL, cache = self.forward_prop(A_prev, parameter["W"+str(L)], parameter["b"+str(L)], "sigmoid" )
        caches.append(cache)
        return AL, caches
    
    
    def backward_prop(self, dZ, cache):
        """
        Input:
        -- dA:
        -- cache: a dictionary contains Z, W, and b for the layer
        Return:
        -- dW: gradient of W
        -- db gradient of db

        """

        ##retrieve parameters from cache
        A_prev, W, b = cache
        dW = (1/m)*np.dot(dZ, A_prev.T)
        db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db
    
    def L_backward_prop(self, AL, caches, Y, layer_sizes):

        L,  m = len(layer_sizes), Y.shape[1]
        cost = self.compute_cost(AL, Y)
        dAL = -np.divide(Y, AL)+ np.divide(1-Y, 1-AL)
        A_prev, dA_prev  = AL, dAL
        grads= {}
        
        dA_prev, cache = self.backward_prop(dA_prev, dAL, caches[L], 'sigmoid', m)
        grads["dW"+str(L)], grads["db"+str(L)]= cache["dW"], cache["db"]

        for l in reversed(range(1,L-1)):
            cache = caches[l]
            dA_prev, cache = self.backward_prop(dA_prev, cache,'relu', m )
            grads["dW"+str(l)]=cache["dW"]
            grads["db"+str(l)]= cache["db"]
        return cost, grads





    def update_parameters(self, W, b, dW, db, learning_rate):
        W -= learning_rate*dW
        b -= learning_rate*db
        return W, b

            
        
        



        



