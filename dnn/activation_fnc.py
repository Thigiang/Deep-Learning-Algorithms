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
        s = 1/(1+np.exp(-Z))
        return s
    def sigmoid_backward(self, dA, Z):
        s = self.sigmoid(Z)
        ds = s*(1-s)
        dZ = dA*ds
        return dZ
    
    def tanh_fnc(self, Z):
        t = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        return t
    def tanh_backward(self, dA, Z):
        t = self.tanh_fnc(Z)
        dt = 1-np.power(t,2)
        dZ = dA*dt
        return dZ
    
    def relu(self, Z):
        r = np.max(0, Z)
        return r
    def relu_backward(self, dA, Z):
        if Z >= 0:
            return dA
        else:
            return 0
    
    def leaky_relu(self, Z):
        lrelu= np.max(0.01*Z, Z)
        return lrelu
    def leaky_relu_backward(self, Z):
        if Z < 0:
            return 0.01*dA
        else: return dA