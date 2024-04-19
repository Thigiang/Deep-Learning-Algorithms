"""
This file create helper functions (activation functions and its derivatives) to support the dnn algorithm (mainly in backward propagation step)

This file provide algorithms to compute different activation functions that used in deep neural networks:
    -- Sigmoid function: s(z) = 1/(1+e^(-z)) and its derivative (called sigmoid_backward function)
    -- tanh funtion: tanh(z) = (e^z - e^(-z))/(e^z + e^(-z)) and its derivative (called tanh_backward function)
    -- ReLU function: relu(z) = max(0,z) and its derivative (called relu_backward)
    -- Leaky ReLU function: lrelu(z) = max(0.01z, z) and its derivative (called Leaky_relu_backward)

"""
import numpy as np
class Activation:
    def sigmoid(self, Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        return A, cache
    
    def sigmoid_backward(self, dA, Z):
        s = self.sigmoid(Z)
        ds = s*(1-s)
        dZ = dA*ds
        return dZ
    
    def tanh_fnc(self, Z):
        A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        cache = Z
        return A, cache
    def tanh_backward(self, dA, Z):
        t = self.tanh_fnc(Z)
        dt = 1-np.power(t,2)
        dZ = dA*dt
        return dZ
    
    def relu(self, Z):
        A = np.max(0, Z)
        cache = Z
        return A, cache
    def relu_backward(self, dA, Z):
        if Z >= 0:
            dZ= dA
            return dZ
        else:
            dZ = 0
            return dZ
    
    def leaky_relu(self, Z):
        A= np.max(0.01*Z, Z)
        cache = Z
        return A, cache
    def leaky_relu_backward(self, Z):
        if Z < 0:
            dZ = 0.01*dA
            return dZ
        else:
            dZ = dA 
            return dZ