"""
This file create helper functions (activation functions and the backward functions to calculate dZ given dA and Z)

This file provide algorithms to compute different activation functions that used in deep neural networks:
    -- Sigmoid function: s(z) = 1/(1+e^(-z)) and sigmoid_backward function: calculating dZ = dA * s'(z)=dA*ds= dA*s(z)*(1-s(z))
    -- tanh funtion: tanh(z) = (e^z - e^(-z))/(e^z + e^(-z)) and tanh_backward function: dZ = dA*dt=dA*(1-t^2)
    -- ReLU function: relu(z) = max(0,z) and relu_backward: dZ = dA*dr where dr is the derivative of relu function: dr=1 if z>00 and dr=0 if z<0
    -- Leaky ReLU function: lrelu(z) = max(0.01z, z) and leaky_relu_backward: dZ = dA*dlr where dlr is the derivative of leaky relu function: dlr = 0.01 if z<0 and dlr=1 if z>=0

"""
import numpy as np
class Activation:
    def sigmoid(self, Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        return A, cache
    
    def sigmoid_backward(self, dA, cache):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA*s*(1-s)
        
        return dZ
    
    def tanh_fnc(self, Z):
        A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        cache = Z
        return A, cache
    def tanh_backward(self, dA, Z):
        t, cache = self.tanh_fnc(Z)
        dt = 1-np.power(t,2)
        dZ = dA*dt
        return dZ
    
    def relu(self, Z):
        A = np.maximum(0, Z)
        cache = Z
        return A, cache
    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z<=0]=0
        return dZ
    def leaky_relu(self, Z):
        A= np.maximum(0.01*Z, Z)
        cache = Z
        return A, cache
    def leaky_relu_backward(self,dA,  Z):
        dZ=np.array(dA, copy=True)
        dZ[Z<=0]=0.01*dA