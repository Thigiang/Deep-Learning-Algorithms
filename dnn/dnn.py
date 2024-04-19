import numpy as np
import copy
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
    
    def linear_forward(self, A, W, b):
        """
        This function computes the linear function: Z = W A +b
        Input:
        -- A: activation from previous layer
        -- W: weight for current layer
        -- b: bias for current layer

        Return:
        -- Z
        -- cache: a tuple that stores A, W, b for backward calcualtion later
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        
        return Z, cache
    

    

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Input:
        -- A_prev: activations from previous layer
        -- W: weight parameter for current layer
        -- b: bias for current layer
        Return:
        -- A: activation for current layer
        -- cache: dictionary contains (Z, (A_prev, W, b)) for current layer
        """
        Z, linear_cache = self.linear_forward(A_prev, W, b) #linear_cache store the input values A_prev, W, b which is needed in backpropagation

        ## Each function in Activation class return the activations for layer l (A) and the value of Z in layer l (activation_cache).
        if activation == 'sigmoid':
            A, activation_cache = Activation.sigmoid(Z)
        elif activation == 'tanh':
            A, activation_cache=Activation.tanh_fnc(Z)
        elif activation == 'relu':
            A, activation_cache = Activation.relu(Z)
        else:
            A, activation_cache = Activation.leaky_relu(Z)
        cache = (linear_cache, activation_cache)

        return A, cache
    
    def L_forward_prop(self, X, parameters):
        """
        This function compute all the activations for all L layers in the neural network. Note that, from layer 1 to L-1 (L-1 layers in total), we use ReLU activation function
        while the last layer L, we use sigmoid activation function (as the last layer is a binary classification)
        Input:
        -- X: A[0] input activation
        -- parameters: initialized parameters. A dictionary that stores the parameters in format {"Wi": Wi, "bi": bi}. Hence, its length is twice
        number of layers.

        Return:
        -- AL: final activation or y hat
        -- caches: a python list that stores every cache of linear_activation_forward. Each element is a tuple (Z, (A_prev, W, b))

        """
        L=len(parameters) // 2
        caches = []
        A = X
        
        for l in range(1, L):
            A_prev = A #The first layer use X as input. A later will be updated as we compute the linear_activation_function()
            W, b = parameters["W"+str(l)], parameters["b"+str(l)] #retrieve the weights and bias for layer l in parameters
            A, cache = self.linear_activation_function(A_prev, W, b, "relu") #A is now updated and used for computing activations of next layer
            caches.append(cache)
        ## At this point, we have computed the activations for the first L-1 layers with relu activation function. We now can compute AL (predicted y) using
        ## sigmoid activation function. Note that A_prev here is the activation of layer L-1
            
        AL, cache = self.linear_activation_forward(A_prev, parameter["W"+str(L)], parameter["b"+str(L)], "sigmoid" )
        caches.append(cache)

        return AL, caches
    
    
    def linear_backward(self, dZ, cache):
        """
        This function compute dA for previous layer (A_prev) and dW, db for current layer l.
        Input:
        -- dA:
        -- cache: a dictionary contains (A_prev, W, b) for the current layer
        Return:
        -- dW: gradient of W
        -- db gradient of db
        """
        ##retrieve parameters from cache
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1/m)*np.dot(dZ, A_prev.T)
        db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db
    
    def linear_activation_backward(self, dA, cache, activation):
        activation_cache, linear_cache = cache
        if activation == 'sigmoid':
            dZ = Activation.sigmoid_backward(dA, activation_cache)
        else:
            dZ = Activation.relu_backward(dA, Z)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db
    
    
    def L_backward_prop(self, AL, Y, caches):

        L, m = len(caches) AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = -np.divide(Y, AL)+ np.divide(1-Y, 1-AL)
        A_prev, dA_prev  = AL, dAL
        grads= {}
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, caches[L-1], 'sigmoid')
        grads["dAL"+str(L-1), ]grads["dW"+str(L)], grads["db"+str(L)]= dAL_prev_temp, dW_temp, db_temp
        for l in reversed(range(1,L-1)):
            cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA_prev_temp, cache, 'relu')
            grads["dAL"+str(l)] = dA_prev_temp
            grads["dW"+str(l+1)]=dW_temp
            grads["db"+str(l+1)]= db_temp
        return grads


    def update_parameters(self, grads, parameters, learning_rate):
        params = copy.deepcopy(parameters)
        L = len(parameters)//2
        for i in range(L):
            params["W"+str(l+1)] =  params["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
            params["b"+str(l+1)] = params["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]
        return params
    
    def dnn_model(self, X, Y, num_iterations, learning_rate, layer_sizes):
        # Initialize parameters
        parameters = self.intitialize_parameters_deep(layer_sizes)
        m = X.shape[0]
        costs = []
        params = parameters
        for i in range(num_iterations):
            AL, caches = self.L_forward_prop(X, params)
            cost = (1/m)*(np.sum(np.dot(Y, np.log(AL))+ np.dot(1-Y, np.log(1-AL))))
            grads = self.L_backward_prop(AL, Y, caches)
            params = self.update_parameters(grads, params, learning_rate)
        return params, costs
    
    def predict(self, X, params):
        m = X.shape[1]
        y_predictions = np.zeros((1, m))
        AL, caches = self.L_forward_prop(X, params)
        for i in range(m):
            if i > 0.5:
                y_predictions[0,i]= 1
            else:
                y_predictions[0,i]= 0
        return y_predictions


        

            
        
        



        



