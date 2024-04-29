import numpy as np
import scipy.io
import sklearn
import sklearn.datasets
import os


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1-s)

    return dZ

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z

    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy = True)
    # print("dA size: ", dA.shape)
    # print("Z size: ", Z.shape)
    # print("dZ size: ", dZ.shape)
    dZ[Z <= 0] = 0

    return dZ

def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples = 300, noise = 0.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples = 100, noise = 0.05)

    return train_X, train_Y, test_X, test_Y
def load_2D_dataset(path):
    file_path = os.path.join(path, 'Datasets/data.mat')
    data = scipy.io.loadmat(file_path)
    train_X = data["X"].T
    train_Y = data["y"].T
    test_X = data["Xval"].T
    test_Y = data["yval"].T

    return train_X, train_Y, test_X, test_Y

def initialize_parameters_zeros(layer_sizes):
    """
    Initialize parameters weights and bias to zeros- Not recommended as it leads to summetry breaking (https://community.deeplearning.ai/t/symmetry-breaking-versus-zero-initialization/16061)

    Arguments:
    layer_sizes -- a Python list stores size (number of units) for all layers (including input layer X)

    Return:
    parameters: a Python dictionary stores the weights and bias for all layers in the network (layer 1 to L)

    """
    L = len(layer_sizes)
    parameters = {}
    for l in range(1, L):
        parameters["W"+str(l)] = np.zeros((layer_sizes[l], layer_sizes[l-1]))
        parameters["b"+str(l)] = np.zeros((layer_sizes[l],1))

    return parameters

def initialize_parameters_random(layer_sizes, scaled):
    """
    Initialize random parameters weights to a scaled value (if scaled is large, weight is large and vice versa) and bias to zeros

    Arguments:
    layer_sizes -- a Python list stores size (number of units) for all layers (including input layer X)
    scaled -- an integer that scales the weights.

    Return:
    parameters: a Python dictionary stores the weights and bias for all layers in the network (layer 1 to L)
    
    """
    np.random.seed(3)
    L = len(layer_sizes)
    parameters = {}

    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1])*scaled
        parameters["b"+str(l)] = np.zeros((layer_sizes[l],1))
    # print(parameters)
    
    return parameters

def initialize_parameters_he(layer_sizes):
    """
    Initialize random parameters weights random values (scaled by a He factor: *np.sqrt(2/size of previous layer)and bias to zeros
    Recommended for ReLU activation

    Arguments:
    layer_sizes -- a Python list stores size (number of units) for all layers (including input layer X)

    Return:
    parameters: a Python dictionary stores the weights and bias for all layers in the network (layer 1 to L)
    
    """
    np.random.seed(1)
    L = len(layer_sizes)
    parameters = {}

    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1])*(np.sqrt(2/layer_sizes[l-1]))
        parameters["b"+str(l)] = np.zeros((layer_sizes[l],1))
    # print(parameters)
    return parameters

def initialize_parameters_xavier(layer_sizes, option = None):
    """
    Initialize random parameters weights random values (scaled by a Xavier initialization factor: *np.sqrt(1/(size of previous layer))and bias to zeros
    Recommended for tanh activation
    Arguments:
    layer_sizes -- a Python list stores size (number of units) for all layers (including input layer X)
    option -- an integer with 1 means using normal Xavier initialization *np.sqrt(2/(dimesnion of previous layer + dimension of current layer)). 
              Defaut is None which uses *np.sqrt(1/dimension of previous layer)

    Return:
    parameters: a Python dictionary stores the weights and bias for all layers in the network (layer 1 to L)
    
    """
    np.random.seed(3)
    L = len(layer_sizes)
    parameters = {}
    if option == 1:
        for l in range(1, L):
            parameters["W"+str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1])*(np.sqrt(2/(layer_sizes[l]+layer_sizes[l-1])))
            parameters["b"+str(l)] = np.zeros((layer_sizes[l],1))
        return parameters

    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1])*(np.sqrt(1/layer_sizes[l-1]))
        parameters["b"+str(l)] = np.zeros((layer_sizes[l],1))
            
    return parameters

def initialize_parameters_deep(layer_sizes, initialization, xavier_option = None, random_scaled=None):
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layer_sizes)
    elif initialization == "random":
        parameters = initialize_parameters_random(layer_sizes, scaled = random_scaled)
    elif initialization == "xavier":
        parameters = initialize_parameters_xavier(layer_sizes, option = xavier_option)
    else:
        parameters = initialize_parameters_he(layer_sizes)  
    return parameters