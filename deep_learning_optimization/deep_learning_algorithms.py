import numpy as np
from dnn_utils import sigmoid, relu, sigmoid_backward, relu_backward,initialize_parameters_deep


"""
---------------------------------------------------------------------------------------------------------------------------------
                                               FORWARD PROPAGATION
"""

def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    else:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    cache = (linear_cache, activation_cache) # ((A_prev, W, b), Z))

    return A, cache

def L_forward(X, parameters):

    L = len(parameters)//2 # Each layer contain 2 parameters Wi and bi in parameters dictionary
    A = X
    caches = []
    """
    From layer 1 to L-1, we want to use relu activation function.
    To compute AL, output layer, we use sigmoid activation function
    """
    ## No dropout
    for l in range(1, L):
        A_prev = A # At the first iteration, A_prev = X is used to compute A1
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b"+str(l)], 'relu')
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches



def dropout(A, keep_prob):
    np.random.seed(1)
    D = np.random.rand(A.shape[0], A.shape[1])
    D = (D < keep_prob).astype(int)
    A = np.multiply(A, D)
    A = A/keep_prob
    return A, D

def L_forward_with_dropout(X, parameters, keep_prob, dropout_layers):

    """
    This function compute the output AL without dop
    """
    np.random.seed(1)
    L = len(parameters)//2 # Each layer contain 2 parameters Wi and bi in parameters dictionary
    A = X
    caches = []
    """
    From layer 1 to L-1, we want to use relu activation function.
    To compute AL, output layer, we use sigmoid activation function
    """
    for l in range(1, L):
        A_prev = A #At first interation, A_prev = X is used to compute A1.
        A_temp, cache_temp = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], 'relu')
        if l in dropout_layers:
            A, D = dropout(A_temp, keep_prob)
            cache = (cache_temp, D)
        else:
            A = A_temp
            cache = (cache_temp, 0)
        caches.append(cache)
    AL, cache_temp = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], 'sigmoid' )
    cache = (cache_temp, 0)
    caches.append(cache)

    return AL, caches


"""
---------------------------------------------------------------------------------------------------------------------------------
                                                  COMPUTE COST
"""

def compute_cost(AL, Y):
    m = Y.shape[1] #number of examples

    log_loss = -np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T)
    # log_loss = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1-AL), 1-Y)
    cost = (1./m) * log_loss
    
    cost = np.squeeze(cost)

    return cost
#Note: With dropout, the cost function is still calculated the same way.

def compute_cost_with_regularization(AL, Y, parameters, lambd):
    m = AL.shape[1]
    L = len(parameters)//2
    log_loss = np.dot(Y, np.log(AL).T)+np.dot(1-Y, np.log(1-AL).T)
    cross_entropy_cost = (-1/m)*np.sum(log_loss)

    L2_regularization_cost = 0
    for l in range(L):
        L2_regularization_cost += (1/m)*(lambd/2)*(np.sum(np.square(parameters["W"+str(l+1)])))

    cost = cross_entropy_cost + L2_regularization_cost
    cost = np.squeeze(cost)

    return cost

def compute_cost_with_dropout(AL, Y):
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1-AL), 1 - Y)
    loss = 1./m * np.nansum(logprobs)

    return loss
"""
---------------------------------------------------------------------------------------------------------------------------------
                                             BACKWARD PROPAGATION
"""


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db
    
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache # A_prev, W, b, Z
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_backpropagation(AL, Y, caches):
    m = AL.shape[1]
    L = len(caches)
    Y = Y.reshape(AL.shape)
    grads = {}
    dAL = -(np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))
    # dA_temp_prev = dAL
    current_cache = caches[L-1]
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL, current_cache, activation = 'sigmoid')

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_temp_prev, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, activation = 'relu' )
        grads["dA"+str(l)] = dA_temp_prev
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp
    return grads

def L_backprop_with_regularization(AL, Y, caches, lambd, parameters):
    
    L, m= len(caches), AL.shape[1]
    Y = Y.reshape(AL.shape)
    grads = {}
    dAL = -np.divide(Y, AL)+np.divide(1-Y, 1-AL)
    # dA_temp_prev = dAL
    dA_temp_prev, dW_temp, db_temp = linear_activation_backward(dAL, caches[L-1], 'sigmoid')
    dW_temp += (lambd/m)*parameters["W"+str(L)]
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = dA_temp_prev, dW_temp, db_temp

    for l in reversed(range(L-1)):
         dA_temp_prev, dW_temp, db_temp = linear_activation_backward(dA_temp_prev, caches[l],'relu' )
         dW_temp += (lambd/m)*parameters["W"+str(l+1)]
         grads["dA"+str(l)], grads["dW"+str(l+1)], grads["db"+str(l+1)] = dA_temp_prev, dW_temp, db_temp
    return grads



def linear_backward_with_dropout(dA, dZ, cache):
    A_prev, W, b = cache # A_prev, W, b
    # print("A prev: ",A_prev.shape)
    # print("W: ", W.shape)
    # print("dZ: ", dZ.shape)
    m = A_prev.shape[1]
    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    # print("dW: ", dW.shape)
    # print("dA_prev: ", dA_prev.shape)
    return dA_prev, dW, db

def L_backprop_with_dropout(AL, Y, caches, keep_prob, dropout_layers):
    """
    caches obtained from L_forward_with_dropout (cache, D) or (cache, 0) where cache = (linear_cache, activation_cache)
    """
    m = AL.shape[1]
    # print(m)
    # print(AL.shape)
    L = len(caches)
    Y = Y.reshape(AL.shape)
    grads = {}

    dAL = (-np.divide(Y, AL) + np.divide((1 - Y), (1 - AL)))
    dZL = AL - Y
    current_cache, D = caches[L-1] #current_cache = cache = linear_cache, activation_cache
    linear_cache, _ = current_cache
    # print(D)
    dA_temp_prev, dW_temp, db_temp = linear_backward_with_dropout(dAL, dZL, linear_cache)
    # print(dA_temp_prev.shape)
    # dA_temp_prev, dW_temp, db_temp = linear_activation_backward_with_dropout(d)

    if L-1 in dropout_layers:
        #if we applied dropout in layer L-1, then we need to do the same for dA L-1
        _ , D_prev = caches[L-2] #retrived D_l-1
        # print("D2: ",  D_prev.shape)
        dA_temp_prev = np.multiply(D_prev, dA_temp_prev)
        dA_temp_prev /= keep_prob
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = dA_temp_prev, dW_temp, db_temp
    # print("dA"+str(L-1)+" size: " + str(grads["dA"+str(L-1)].shape))

    for l in reversed(range(L-1)):
        # print(l)
        A, _, _ = linear_cache
        dZ = np.multiply(dA_temp_prev, np.int64(A > 0))
        current_cache, D = caches[l]  
        linear_cache, _ = current_cache
        # print(D)
        # print(current_cache)
        if l in dropout_layers: #check if we had applied dropout for layer l. If yes, we need to do the same during backprop. Otherwise, just compute dA_temp_prev, dW_temp, db_temp as usual
            """
            We had shutdown some neuron and divided dA_l in forward propagation by keep_prob, in backward propagation,
            we need to do the same thing before we compute dA_l-1 and dWl, dbl
            """
            dA_temp_prev, dW_temp, db_temp = linear_backward_with_dropout(dA_temp_prev,dZ, linear_cache)
            _ , D_prev = caches[l-1]  #we need D_prev to shutdown the same neurons and divived A_temp_prev by keep_prob as we did in forwardprop
            dA_temp_prev = np.multiply(D_prev, dA_temp_prev)
            dA_temp_prev /= keep_prob      
        else:
            dA_temp_prev, dW_temp, db_temp = linear_backward_with_dropout(dA_temp_prev,dZ,  linear_cache)
        grads["dA"+str(l)], grads["dW"+str(l+1)], grads["db"+str(l+1)] = dA_temp_prev, dW_temp, db_temp
        # print("dA"+str(l)+" size: " + str(grads["dA"+str(l-1)].shape))

    return grads


"""

---------------------------------------------------------------------------------------------------------------------------------

                                            GRADIENT CHECKING
                                            
"""

def dictionary_to_vector(parameters):
    keys, layer_sizes = [], []
    count = 0
    params_keys = list(parameters.keys())
    # print(params_keys)
    for key in params_keys:
        if key[0]=="W":
            layer_sizes.append(parameters[key].shape[1])
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        # print("parameters count: ", count, len(new_vector))
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis =0)
        count = count + 1
    layer_sizes.append(parameters[key].shape[0])
    cache = (keys, params_keys, layer_sizes)
    return theta, cache


def vector_to_dictionary(theta, keys, layer_sizes):
    parameters = {}
    key, count = np.unique(keys, return_counts=True)
    key_count = dict(zip(key, count))
    params=[]
    for i in range(1, len(layer_sizes)):
        params.append("W"+str(i))
        params.append("b"+str(i))
    start = 0
    for l in range(1, len(params),2):
        #l=1 params[0], params[1]
        #l=2 params[1], params[2]
        count_w, count_b = key_count[params[l-1]], key_count[params[l]]
        parameters["W"+str(l//2+1)] = theta[start:start+count_w].reshape((layer_sizes[l//2+1], layer_sizes[l//2]))
        start += count_w
        parameters["b"+str(l//2+1)] = theta[start:start+count_b].reshape((layer_sizes[l//2+1],1))
        start += count_b
    return parameters

def gradients_to_vector(gradients, params_keys):
    count = 0
    grads_keys = ["d"+i for i in params_keys]
    # print(grads_keys)
    for key in grads_keys:
        # print(key)
        new_vector = np.reshape(gradients[key], (-1,1))
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count += 1
    return theta

def gradients_check(parameters, gradients, X, Y, epsilon = 1e-7, print_msg=False):

    parameters_values, cache = dictionary_to_vector(parameters)
    keys, params_keys, layer_sizes = cache
    grad = gradients_to_vector(gradients, params_keys)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters,1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters,1))
    

    for i in range(num_parameters):

        theta_plus = np.copy(parameters_values)
        theta_plus[i] = theta_plus[i] + epsilon
        AL, _ = L_forward(X, vector_to_dictionary(theta_plus, keys, layer_sizes))
        J_plus[i] = compute_cost(AL, Y)

        theta_minus = np.copy(parameters_values)
        theta_minus[i]=theta_minus[i] - epsilon
        AL, _ = L_forward(X, vector_to_dictionary(theta_minus, keys, layer_sizes))
        J_minus[i] = compute_cost(AL, Y)

        gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator/denominator

    if print_msg:
        if difference > 2e-7:
            print("\033[93m" + "There is a mistake in the backward propagation! difference = "+ str(difference))
        else:
            print("\033[92m" + "Your backward propagation works perfectly fine! difference = "+ str(difference))
    return difference



"""
---------------------------------------------------------------------------------------------------------------------------------
                                             UPDATE PARAMETERS
"""


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2
    for l in range(L):
        parameters["W"+str(l+1)] -= learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] -= learning_rate*grads["db"+str(l+1)]
    return parameters



"""
---------------------------------------------------------------------------------------------------------------------------------
                                        NEURAL NERWORK MODEL WITH L LAYERS
"""

def deep_neural_net_opt(X, Y, initialization, layer_sizes, learning_rate=0.075, num_iterations=300, 
                        lambd=0.0, keep_prob = 1.0, dropout_layers = None,random_scaled= None, 
                        xavier_option = None, print_cost = False, grad_check=False):

    """
    Step 1: Initializae parameters
    Step 2: Use forward propagation to compute AL
    Step 3: Compute cost using AL, and Y
    Step 4: Use backward propagation to compute derivatives
    Step 5: Update parameters
    
    """
   # Step 1: Initialize parameters. 

    if initialization == "zeros":
        parameters = initialize_parameters_deep(layer_sizes, "zeros")
    elif initialization == 'random':
        parameters = initialize_parameters_deep(layer_sizes, 'random', random_scaled = random_scaled)
    elif initialization == 'xavier':
        parameters = initialize_parameters_deep(layer_sizes, 'xavier', xavier_option = xavier_option)
    else:
        parameters = initialize_parameters_deep(layer_sizes, 'he')
        
    # Step 2: Use forward propagation to compute derivatives
    # Case 1: No regularization
    costs = []
    if lambd == 0. and keep_prob == 1.0:
        for i in range(num_iterations):
            AL, caches = L_forward(X, parameters)  #caches stores (linear_cache, activation_cache) for all layers
            
            cost = compute_cost(AL, Y)
            grads = L_backpropagation(AL, Y, caches)
            if grad_check:
                difference = gradients_check(parameters, grads, X, Y, 1e-7, print_msg=True)
            parameters = update_parameters(parameters, grads, learning_rate = learning_rate)
            if i % 1000 == 0:
                if print_cost:
                    print("Cost at {}th iteration is: {}".format(i, np.squeeze(cost)))
                costs.append(cost)
            
    #Case 2: regularization using dropout
    if keep_prob < 1.0:
        for i in range(num_iterations):
            AL, caches = L_forward_with_dropout(X, parameters, keep_prob = keep_prob, dropout_layers = dropout_layers)
            # caches stores ((linear_cache, activation_cache), D) for layers in dropout_layers and ((linear_cache, activation_cache), 0) for layers without dropout
            cost = compute_cost_with_dropout(AL, Y)
            grads = L_backprop_with_dropout(AL, Y, caches, keep_prob=keep_prob, dropout_layers=dropout_layers)
            parameters = update_parameters(parameters, grads, learning_rate = learning_rate)
            if i % 1000 == 0:
                if print_cost:
                    print("Cost at {}th iteration is: {}".format(i, np.squeeze(cost)))
                costs.append(cost)
    # Case 3: regularization using L2
    if lambd != 0:
        for i in range(num_iterations):
            AL, caches = L_forward(X, parameters)
            cost = compute_cost_with_regularization(AL, Y, parameters, lambd = lambd)
            grads = L_backprop_with_regularization(AL, Y, caches, lambd=lambd, parameters=parameters)
            parameters = update_parameters(parameters, grads, learning_rate=learning_rate)
            if i % 1000 == 0:
                if print_cost:
                    print("Cost at {}th iteration is: {}".format(i, np.squeeze(cost)))
                costs.append(cost)
    return parameters, grads, costs

def predict( X, Y, params):
        #We don't use dropout during testing
        """
        This function is used to predict the label of a new dataset-different to train_X-using the learned parameters.

        Arguments:
        -- X: a new dataset (or test dataset) that we want to predict the labels
        -- params: a dictionary that stores the learned parameters using gradient descent

        Return:
        -- y_predictions: A matrix of size (1, m) that stores the predicted labels for the dataset X.

        """
        m = X.shape[1]
        y_predictions = np.zeros((1, m))
        AL, caches = L_forward(X, params)
        for i in range(AL.shape[1]):
            if AL[0,i] > 0.5:
                y_predictions[0,i]= 1
            else:
                y_predictions[0,i]= 0
        accuracy = np.sum((y_predictions==Y)/Y.shape[1])
        return y_predictions, accuracy
    
    