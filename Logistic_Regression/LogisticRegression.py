import numpy as np
import copy
class LogReg:  
    def sigmoid(self, z):
        """
        Arguments: z -- a scalar or numpy array of any size
        Return:  s -- sigmoid function of z
        """
        s = 1/(1+np.exp(-z))
        return s
    def initialize_with_zeros(self, dim):
        """
        This function initializes a vector zeros of shape (dim, 1) for w and 0 for b
        Arguments: dim -- the first dimension of b which is the number of parameters in X
        Return: w -- initialized vector w of shape (dim, 1) and b -- initialized scalar.
        
        """
        w = np.zeros((dim, 1))
        b = 0.
        return w, b
    def propagation(self, w, b, X, Y):
        """
        This function implements the cost function and its gradient.
        Arguments:
        w -- weights, a numpy array of size (num_px*num_px*3, 1)
        b -- a scalar, bias
        X -- a matrix of size (num_px*num_px*3, number of examples)
        Y -- a numpy array of size (1, number of examples)-it contains true labels
        Returns:
        grads -- a dictionary contain the gradients of weights (dw) and bias (db)
        cost -- negative log-likelihood cost for logistic regression
        """
        # FORWARD PROPAGATION FROM X TO COST
        m = X.shape[1] #number of examples (images)
        #Compute activation
        A = self.sigmoid(np.dot(w.T, X)+b)
        #Compute the cost function
        cost = (-1/m)*np.sum(np.dot(Y, np.log(A).T)+ np.dot((1-Y), np.log(1-A).T))
        
        #BACK WARD PROPAGATION FROM dz to dw, db (gradients of weights and bias)
        dw = (1/m)*np.dot(X, (A-Y).T)
        db = (1/m)*np.sum(A-Y)
        
        cost = np.squeeze(np.array(cost))
        grads = {"dw": dw, "db": db}
        return grads, cost
    
    def optimize(self, w, b, X, Y, num_iterations = 1000, learning_rate = 0.005, print_cost=False):
        """
        This function optimize the logistic regression performance by minimizing the cost function. It optimizes
        w and b by running the gradient descent algorithm.
        Arguments:
        w -- weights a numpy array of size (num_px*num_px*3, 1)
        b -- bias, a scalar
        X -- training data, a matrix of size (num_px*num_px*3, number of examples)
        Y -- a numpy array of size (1, number of examples) that contains true labels for the training data.
        num_iterations -- number of interations that we want to run to train w and b
        learning_rate -- how big of a step we want w and b to learn in each iteration
        print_cost -- True to print the cost every 100 steps.
        
        Returns:
        params: a dictionary that stores the wights-w and bias-b
        grads: a dictionary contain the gradients of weights (dw) and bias (db)
        costs: list of all the costs computed during the optimization.
        
        """
        w, b, costs = copy.deepcopy(w), copy.deepcopy(b), []
        
        for i in range(num_iterations):
            
            grads, cost = self.propagation(w, b, X, Y)
            
            #Retrieve the derivatives of w and b from grads
            dw, db = grads["dw"], grads["db"]
            
            #Use update rule to update w and b
            w -= learning_rate*dw
            b-= learning_rate*db
            
            #print cost every 100 iterations
            if i%100 == 0:
                costs.append(cost)
                if print_cost:
                    print("Cost after interation {}: {}".format(i, cost))
                    
        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}
        return params, grads, costs
    
    def predict(self, w, b, X):
        """
        This function predicts the labels of examples in training data X using
        learned logistic regression parameters w and b
        
        Arguments:
        w -- a numpy array of size (num_px*num_px*3, 1)
        b -- a scalar
        X -- data of size (num_px*num_px*3, number of examples)
        
        Returns:
        Y_prediction: a numpy array of size (1, number of examples) that contains the predicted labels for data X.
        """
        
        m=X.shape[1] # number of examples in data X
        
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)
        A = self.sigmoid(np.dot(w.T,X)+b)
        #Convert probabilities A[0,i] to actual predictions
        for i in range(A.shape[1]):
            if A[0,i]> 0.5:
                Y_prediction[0,i]=1
            else:
                Y_prediction[0,1]=0
        return Y_prediction
    
    def model(self, X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate = 0.005, print_cost=False):
        """
        This function calls the above implemented functions to build a logistic regression model.
        
        Arguments:
        X_train -- training data of size (num_px*num_px*3, number of example in X_train)
        Y_train -- a numpy array of size (number of examples in X_train, 1) contains true labels.
        X_test -- testing data of size (num_px*num_px*3, number of examples in X_test)
        Y_test -- a numpy array of size (number of examples in X_test, 1) contains true labels.
        num_iterations -- hyperparameter representing the number of interations to optimize w and b
        learning_rate -- hyperparamter representing the learning rate used in the update rule of optimize()
        print_cost -- True to print the cost every 100 steps and the train and accuracy rate.
        
        Returns:
        d -- a dictionary containing information about the model
        
        """
        #Initialize parameters w and b with zeros
        w, b = self.initialize_with_zeros(X_train.shape[0])
        
        #optimize the parameters using gradient descent
        params, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations=num_iterations,
                                            learning_rate = learning_rate, print_cost = print_cost)
        
        # Retrieve the weights and bias from params
        w, b = params["w"], params["b"]
        
        #Predict the labels using learned logistic regression parameters
        Y_prediction_test=self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)
        
        #Print the train and test accuracy if print_cost is True
        if print_cost:
            print("train accuracy: {} %".format(100-np.mean(abs(Y_prediction_train-Y_train))*100))
            print("test accuracy: {} %".format(100-np.mean(abs(Y_prediction_test-Y_test))*100))
            
        d={"costs":costs,
          "Y_prediction_test": Y_prediction_test,
          "Y_prediction_train": Y_prediction_train,
          "w": w, "b":b,
          "learning_rate": learning_rate,
          "num_iterations": num_iterations}
        return d
        
            
        
        
        