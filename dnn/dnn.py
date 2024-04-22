import numpy as np
import copy
from activation_fnc import Activation
Activation = Activation()
class dnn:
    def initialize_parameters_deep(self, layer_sizes):
        """
        This function initialize weights and bias paramters for deep learning algorithm.
        Arguments:
         -- layer_sizes: a list of array that contains the size of all the layers (including the input layer n_x or n_0)
        
        Return:
        -- parameters: a dictionary with initialized parameters for all the layers.
        """
        np.random.seed(1) 
        L = len(layer_sizes)
        parameters = {}

        for l in range(1, L):
            n_x, n_h= layer_sizes[l-1], layer_sizes[l]
            parameters["W"+str(l)]= np.random.randn(n_h, n_x)/np.sqrt(layer_sizes[l-1])
            parameters["b"+str(l)]= np.zeros((n_h, 1))

        return parameters
    
    def linear_forward(self, A, W, b):
        """
        This function computes the linear function: Z = W A +b

        Arguments:
        -- A: a matrix activation for previous layer which has size 
        -- W: a matrix weight for current layer 
        -- b: a column vector bias for current layer

        Return:
        -- Z: a matrix with the same size as A
        -- cache: a tuple that stores input arguments (A, W, b)- this is useful for backward calcualtion later
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        This function compute the activation A for current layer given activation from previous layer, weight and bias for current layer.

        Arguments:
        -- A_prev: A matrix activations from previous layer
        -- W: A matrix weight parameter for current layer 
        -- b: A column vector bias for current layer

        Return:
        -- A: A matrix activation for current layer 
        -- cache: dictionary contains (Z, (A_prev, W, b)) where Z is calculated in the function and (A_prev, W, b) are input arguments
        """

        Z, linear_cache = self.linear_forward(A_prev, W, b) 
        #Recall that linear_cache store the input arguments (A_prev, W, b)

        ## Recall: Each function in Activation class return the activations (A) and the input argument of Z (cache) for current layer l
        if activation == 'sigmoid':
            A, activation_cache = Activation.sigmoid(Z)
        # elif activation == 'tanh':
        #     A, activation_cache=Activation.tanh_fnc(Z)
        elif activation == 'relu':
            A, activation_cache = Activation.relu(Z)
        # else:
        #     A, activation_cache = Activation.leaky_relu(Z)

        cache = (linear_cache, activation_cache)

        return A, cache
    
    def L_forward_prop(self, X, parameters):
        """
        This function compute the activations for all L layers in the neural network. Note that, from layer 1 to L-1 (L-1 layers in total), we use ReLU activation function
        while the last layer L, we use sigmoid activation function (as the last layer is a binary classification)

        Arguments:
        -- X: A^[0] a matrix X with size
        -- parameters: initialized parameters. A dictionary that stores the parameters in format {"Wi": Wi, "bi": bi}. Hence, its length is twice
        number of layers.

        Return:
        -- AL: final layer activation or y hat
        -- caches: a python list that stores every cache of linear_activation_forward. Each element is a tuple (Z, (A_prev, W, b))

        """

        L=len(parameters) // 2
        caches = []
        A = X
        
        for l in range(1, L):
            A_prev=A #The first layer use X as input. A later will be updated as we compute the linear_activation_function()
            W, b = parameters["W"+str(l)], parameters["b"+str(l)] #retrieve the weights and bias for layer l from parameters
            A, cache = self.linear_activation_forward(A_prev, W, b, "relu") #A is now updated and used as A_prev for computing next layer activations 
            caches.append(cache)

        """
        At this point, we have computed the activations for the first L-1 layers with relu activation function. We now can compute AL (predicted y) using
        sigmoid activation function. Note that A_prev here is the layer (L-1)th activation      
        """      
        AL, cache = self.linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid" )
        caches.append(cache)

        return AL, caches
    
    def compute_cost(self, AL, Y):
        """
        This function computes the cost of the algorithm after getting AL or y_hat

        Arguments:
        -- AL: final output (output of the final layer) or predicted Y
        -- Y: true labels 
        Return:
        -- cost: the cost of the algorithm

        """
        m = Y.shape[1]
        # loss = np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL), (1-Y))
        # cost = (-1/m)*np.sum(loss)
        cost = (1./m)*(-np.dot(Y, np.log(AL).T)- np.dot((1-Y), np.log(1-AL).T))
        cost = np.squeeze(cost)
        return cost
    
    def linear_backward(self, dZ, cache):
        """
        This function computes dA for previous layer l-1 and dW, db for current layer l assuming that we have gotten dZ.

        Arguments:
        -- dZ: a matrix with same size as Z
        -- cache: contains (A_prev, W, b)

        Return:
        -- dA_prev: matrix with same size as A_prev
        -- dW: gradient of W in layer l 
        -- db: gradient of db in layer l
        """
        ##retrieve parameters from cache
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1./m)*np.dot(dZ, A_prev.T)
        db = (1./m)*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db
    
    def linear_activation_backward(self, dA, cache, activation):
        """
        This function computes dZ using activation backward functions, and then computes dA_prev, dW, db using linear_backward() function

        Arguments:
        -- dA: a matrix with same dimension as matrix A (activation for current layer l)
        -- cache: a tuple contains activation_cache (stores Z) and linear_cache (stores A_prev, W, b)
        -- activation: a string to indicate which activation backward function to use

        Return:
        -- dA_prev: a matrix with same dimension as matrix A_prev (activation for previous layer l-1)
        -- dW: gradient of W
        -- db: gradient of b
        """
        linear_cache, activation_cache = cache
        if activation == 'sigmoid':
            dZ = Activation.sigmoid_backward(dA, activation_cache)
        elif activation == "relu":
            dZ = Activation.relu_backward(dA, activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db
    
    
    def L_backward_prop(self, AL, Y, caches):

        """
        This function does the same job as linear_activation_backward() function, but for all the layers in neural networks starting from layer l-1.
        Given AL-the final output or y_hat, we will compute dAL. dAL then will be treated as an input argument for function linear_activation_backward()
        to calculate dA_prev, dW and db.

        Arguments:
        -- AL- final output (or y_hat)
        -- Y: the true labels
        -- caches: a Python list that stores the (activation_cache, linear_cache) for all the layers in the neural networks

        Return:
        -- grads: a dictionary that stores the gradients dA_prev, dW, db for all the layers.
        
        """

        L, m = len(caches), AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = -np.divide(Y, AL)+ np.divide((1-Y), (1-AL))
        # A_prev, dA_prev  = AL, dAL
        grads= {}
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, caches[L-1], 'sigmoid')
        grads["dAL"+str(L-1)]= dA_prev_temp
        grads["dW"+str(L)]=dW_temp
        grads["db"+str(L)]=db_temp
        for l in reversed(range(L-1)):
            cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA_prev_temp, cache, 'relu')
            grads["dAL"+str(l)] = dA_prev_temp
            grads["dW"+str(l+1)]=dW_temp
            grads["db"+str(l+1)]= db_temp

        return grads


    def update_parameters(self, grads, params, learning_rate):
        """
        This function updates the parameters W, b after getting the gradients

        Arguments:
        -- grads: a dictionary contains gradients dA_prev, dW, db
        -- parameters: a dictionary stores the initialized parameters W, b
        -- learning_rate: a floating number that determines how big a step we want the parameters to be updated

        Return:
        -- params: a dictionary contains the learned parameters using gradient descent.
        
        """
        parameters = copy.deepcopy(params)
        L = len(parameters)//2
        for l in range(L):
            parameters["W"+str(l+1)] =  parameters["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
            parameters["b"+str(l+1)] =  parameters["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]
        return parameters
    
    
    def fit(self, X, Y, num_iterations, learning_rate, layer_sizes, print_cost = False):
        """
        This function aims to combine all the functions we have built earlier to make a deep neural networks model.
        
        Arguments:
        -- X: training data
        -- Y: true labels for training data
        -- num_interations: an integer number of iterations we want to train the parameters.
        -- learning_rateL a floating number that defines how big a step we want the parameters to be updated at each iteration
        -- layer_sizes: a Python lists that contains the size of the layers (including the input layer-X)

        Return:
        -- params: a dictionary that stores the learned parameters using gradient descent
        -- costs: a Python list that contains the cost of the leraning algorithm at each iteration.
        """
        # Initialize parameters
        # np.random.seed(13)
        parameters = self.initialize_parameters_deep(layer_sizes)
        costs = []
        for i in range(0,num_iterations):
            AL, caches = self.L_forward_prop(X, parameters)
            cost = self.compute_cost(AL, Y)
            grads = self.L_backward_prop(AL, Y, caches)
            parameters = self.update_parameters(grads, parameters, learning_rate)
            if i%100==0 or i == num_iterations-1:
                costs.append(cost)
                if print_cost:
                    print("Cost at {}th iteration is: {}.".format(i, np.squeeze(cost)))
        return parameters, costs
    
    def predict(self, X, params):
        """
        This function is used to predict the label of a new dataset using the learned parameters.

        Arguments:
        -- X: a new dataset (or test dataset) that we want to predict the labels
        -- params: a dictionary that stores the learned parameters using gradient descent

        Return:
        -- y_predictions: A matrix of size (1, m) that stores the predicted labels for the dataset X.

        """
        m = X.shape[1]
        y_predictions = np.zeros((1, m))
        AL, caches = self.L_forward_prop(X, params)
        for i in range(AL.shape[1]):
            if AL[0,i] > 0.5:
                y_predictions[0,i]= 1
            else:
                y_predictions[0,i]= 0

    def accuracy(self, Y_pred, Y):
        accuracy = np.sum((Y_pred==Y)/Y.shape[1])
        return accuracy

        

            
        
        



        



