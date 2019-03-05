"""
Author: Marnie Biando
Filename: Deep_NN_MBiando.py

Code submitted for Neural Networks Class, Project #1
Build Deep Neural Net from Three-Layer Neural Net

Note: uses the three_layer_NN which does not work in any virtualenv
NOT SURE WHY

"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt



def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()



########################################################################################################################
# Start coding here.
########################################################################################################################

class deep_NN(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, input_layer, output_layer, actFun_type, seed=10):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.actFun_type = actFun_type

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W = np.random.randn(self.input_layer, self.output_layer) / np.sqrt(self.input_layer)
        self.b = np.zeros((1, self.output_layer))


    def actFun(self, a, non_Linearity):
        '''
        actFun computes the activation functions
        :param a = net input
        :param non_Linearity = Tanh, Sigmoid, or ReLU
        :return: net activation
        '''

        # YOU IMPLMENT YOUR actFun HERE
        if non_Linearity == "tanh":
            return np.tanh(a)
        
        elif non_Linearity == "reLU":
            return np.maximum(0,a)    
        
        elif non_Linearity == "sigmoid":
            return 1/(1+np.exp(-a))               


    def diff_actFun(self, a, non_Linearity):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param a= net input
        :param non_Linearity = Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        if non_Linearity == "tanh":
            return 1-np.power(np.tanh(a),2)
        
        elif non_Linearity == "reLU":
#            return np.heaviside(a, 0)
            a[a<=0] = 0
            a[a>0] = 1
            return a
        
        elif non_Linearity == "sigmoid":
            return (1/(1+np.exp(-a))) * (1-(1/(1+np.exp(-a))))


    def ForwardPass(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return: None??? or output from softmax function?
        '''

        # YOU IMPLEMENT YOUR ForwardPass HERE
        X1_ones = np.ones((len(X), 1))
        X1 = np.hstack((X, X1_ones))
        W1 = np.vstack((self.W, self.b))

        self.a1 = X1.dot(W1)
        self.act1 = actFun(self.a1)

# The next line is only performed in last layer  
#     self.probs = self.softmax(self.a2)

        return self.a1, self.act1


    def BackwardPass(self, prev_delta, prev_W, curr_X, curr_a, actFun):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW, dL/b
        '''
        
        # IMPLEMENT YOUR BACKPROP HERE
        curr_delta = np.dot(prev_delta, prev_W.T) * self.diff_actFun(curr_a, self.actFun_type)
        dW = np.dot(curr_X.T, curr_delta)
        db = np.sum(curr_delta, axis=0, keepdims=True)
            
        return dW, db, curr_delta

class Layers(object):
    """
    This class takes a list containing information on number of layers in Neural Network
    and uses that information to generate a list of dictionaries, with each layer
    represented as a dictionary
    """

    def __init__(self, architecture, t, reg_lambda=0.01, seed=10):
        '''
        :input: architecture, which is a list of number of neurons at each layer
        :track_layers: initiate W and b matrices for each layer
        :forwardpass(): take in X and run deep_nn.fit_model()
        :output: <update>
        '''
        
        # keep neural net architecture for reference/use
        self.architecture = architecture
        self.t = t
        self.reg_lambda = reg_lambda
        # nn_params will track values for all layers:
        # X, W, b, a(W*X+b), act(actFun(a))
        # W and b will be updated during backpropagation
        self.nn_params = []
        
        
    def track_layers(self):
        '''
        :input: none
        :initializes weights and bias matrices for each layer
        :output: foward_vals, a list containing dictionaries for each layer
        '''
        for i in range(len(self.architecture)):
            # self.nn_params belongs to Layers class
            # each dictionary represents layers variables (W, b, a, act, etc)
            # here we create a deep_NN object and strore in the dictionary
            this_layer = deep_NN(self.architecture[i].get("input"),
                    self.architecture[i].get("output"),
                    self.architecture[i].get("activation"))
            this_dict = {"index": i+1,
                         "layer": this_layer, 
                         "actFun": self.architecture[i].get("activation"),
                         "W": this_layer.W,
                         "b": this_layer.b}
            self.nn_params.append(this_dict)            

        return(self.nn_params)    

    def softmax(self, z):

        exp_sum = np.sum(np.exp(z), 1)
        t = np.tile(exp_sum, [2, 1]).T
        out = np.exp(z) * (1 / t)
        return out

    def calculate_loss(self, X, t):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        data_loss =(1/num_examples)* np.sum((-t*np.log(self.probs)))
        self.losses.append(data_loss)
        return(np.nan_to_num(data_loss))


    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        
#        self.ForwardPass(X, lambda x: self.actFun(x, self.actFun_type))
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def feedforward(self, X):
        '''
        : loops through nn_params, and for each layer, calls ForwardPass,
        : calculating W*X+b=a and actFun(a)
        '''        
        self.nn_params[0]['X'] = X
        
        for i in range(len(self.nn_params)):
            # access correct deep_NN object from nn_params
            this_layer = self.nn_params[i].get('layer')
            # retrieve correct X to pass along to ForwardPass            
            X = self.nn_params[i].get('X')
            # call ForwardPass function for current deep_NN object 
            actFun_type = self.nn_params[i].get('actFun')

            a1, act1 = this_layer.ForwardPass(X, lambda x: this_layer.actFun(x, actFun_type))              
            self.nn_params[i]['a'] = a1
            self.nn_params[i]['act'] = act1
            if i < len(self.nn_params)-1:
                self.nn_params[i+1]['X'] = act1
            # at last layer, calculate the y_hat values:    
            if i == len(self.nn_params)-1:
                self.probs = self.softmax(a1)
                self.nn_params[i]['act'] = self.probs
        
        return None

    def backprop(self, epsilon):
        '''
        : loops through nn_params, and for each layer, calculate dW and db,
        : these values are then used to update W and b at each layer
        '''                
        for i in range(len(self.nn_params) - 1, -1, -1):
            # access correct deep_NN object from nn_params
            this_layer = self.nn_params[i].get('layer')
            # retrieve W and b values needed... prev = layer AFTER current layer
            curr_W = self.nn_params[i].get('W')
            curr_X = self.nn_params[i].get('X')
            curr_a = self.nn_params[i].get('a')
            actFun_type = self.nn_params[i].get('actFun') 

            # don't call backprop for last layer
            if i == len(self.nn_params)-1: 
                self.prev_delta = self.probs - self.t
                db = np.sum(self.prev_delta, axis=0, keepdims=True)
                dW = self.nn_params[i].get('X').T.dot(self.prev_delta)
                
            # all other layers: 
            # def BackwardPass(self, prev_delta, prev_W, curr_X, curr_a, actFun)
            else:
                prev_W = self.nn_params[i+1].get('W')
                dW, db, self.prev_delta = this_layer.BackwardPass(self.prev_delta, prev_W, curr_X, curr_a, lambda x: this_layer.actFun(x, actFun_type))
            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW += self.reg_lambda * curr_W

            # Gradient descent parameter update
            self.nn_params[i]['W'] += -epsilon * dW
            self.nn_params[i]['b'] += -epsilon * db

    def fit_model(self, X, t, epsilon, num_passes=15000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        
        self.losses = []
        
        # Gradient descent implementation
        for i in range(0, num_passes):
            # Layers.feedforward calls deep_nn.forwardPass()
            self.feedforward(X)
            # Layers.backprop calls deep_nn.BackwardPass()
            self.backprop(epsilon)

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:

                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, t)))
                result = self.calculate_loss(X, t)
                self.losses.append(result)

        mean = np.mean(self.losses)

        return mean, self.losses

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    # # generate and visualize Make-Moons dataset
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.25)
    t = np.zeros((len(y), 2))
    for i, val in np.ndenumerate(y):
        if val == 0:
            t[i, 0] = 0
            t[i, 1] = 1
        else:
            t[i, 0] = 1
            t[i, 1] = 0
    plt.scatter(X[:, 0], X[:, 1], s=45, c=y, cmap=plt.cm.plasma)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.show()

# SHALLOW network (1 hidden layer) -- loss: 3.15e-2
#    architecture = [ 
#            {"input": 2, "output": 100, "activation": "tanh"},
#            {"input": 100, "output": 2, "activation": "tanh"}]

# DEEP NETWORK (4 hidden layers) -- smallest loss: 3.71e-4    
#    architecture = [ 
#        {"input": 2, "output": 20, "activation": "tanh"},
#        {"input": 20, "output": 20, "activation": "tanh"},
#        {"input": 20, "output": 20, "activation": "tanh"},
#        {"input": 20, "output": 20, "activation": "tanh"},
#        {"input": 20, "output": 2, "activation": "tanh"}]
    
# IN THE MIDDLE (2 hidden layers) -- loss: 2.65e-3
#    architecture = [ 
#        {"input": 2, "output": 20, "activation": "tanh"},
#        {"input": 20, "output": 20, "activation": "tanh"},
#        {"input": 20, "output": 2, "activation": "tanh"}]    

# DEEP NETWORK (4 hidden layers) -- loss: 8.42e-2   
#    architecture = [ 
#        {"input": 2, "output": 20, "activation": "sigmoid"},
#        {"input": 20, "output": 20, "activation": "sigmoid"},
#        {"input": 20, "output": 20, "activation": "sigmoid"},
#        {"input": 20, "output": 20, "activation": "sigmoid"},
#        {"input": 20, "output": 2, "activation": "sigmoid"}]   

# DEEP NETWORK (similar to above, but swapped tanh for sigmoid)   
    architecture = [ 
        {"input": 2, "output": 20, "activation": "sigmoid"},
        {"input": 20, "output": 20, "activation": "tanh"},
        {"input": 20, "output": 20, "activation": "sigmoid"},
        {"input": 20, "output": 20, "activation": "tanh"},
        {"input": 20, "output": 20, "activation": "sigmoid"},
        {"input": 20, "output": 2, "activation": "sigmoid"}]
    
#    architecture = [ 
#        {"input": 2, "output": 100, "activation": "sigmoid"},
#        {"input": 100, "output": 2, "activation": "sigmoid"}]    


# reLU doesn't work -- I think I know why but can't fix it    
#    architecture = [ 
#        {"input": 2, "output": 3, "activation": "tanh"},
#        {"input": 3, "output": 50, "activation": "sigmoid"},
#        {"input": 50, "output": 5, "activation": "reLU"},
#        {"input": 5, "output": 2, "activation": "tanh"}]


# Create model by calling Layer class which will call deep_NN class    
    model = Layers(architecture, t)
    model.track_layers()
    
# Run the fit_model with varying learning size:
    epsilons = [0.001, 0.0001, 0.00001]    
#    epsilons = [0.005]
    
    track_loss = {}
    
    for epsilon in epsilons:
        print("Testing with epsilon =", epsilon)
        mean, loss = model.fit_model(X, t, epsilon)
        track_loss[epsilon] = loss
        model.visualize_decision_boundary(X, y)

    x = np.arange(len(loss))
    plt.plot(x, track_loss[0.001])
    plt.plot(x, track_loss[0.0001])
    plt.plot(x, track_loss[0.00001])
    plt.title("Avg Loss per Iteration")
    plt.xlabel("45 values per Iteration")
    plt.ylabel("Avg Loss")
    plt.legend(['epsilon = 0.001', 'epsilon = 0.0001', 'epsilon = 0.00001'], loc='upper right')

    plt.show()
    



if __name__ == "__main__":
    main()


