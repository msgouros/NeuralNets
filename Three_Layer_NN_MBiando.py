"""
Author: Marnie Biando
Filename: breast_cancer_classification.py

Note: uses the three_layer_NN which does not work in virtualenv

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

class three_layer_NN(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, input_layer, hidden_layer, output_layer, actFun_type='tanh', reg_lambda=0.01, seed=10):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.input_layer, self.hidden_layer) / np.sqrt(self.input_layer)
        self.b1 = np.zeros((1, self.hidden_layer))
        self.W2 = np.random.randn(self.hidden_layer, self.output_layer) / np.sqrt(self.hidden_layer)
        self.b2 = np.zeros((1, self.output_layer))


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


    def softmax(self, z):

        exp_sum = np.sum(np.exp(z), 1)
        t = np.tile(exp_sum, [2, 1]).T
        out = np.exp(z) * (1 / t)

        return out

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
        W1 = np.vstack((self.W1, self.b1))
        self.a1 = X1.dot(W1)
        self.act1 = actFun(self.a1)
        act1_ones = np.ones((len(self.act1), 1))
        act1 = np.hstack((self.act1, act1_ones))
        W2 = np.vstack((self.W2, self.b2))
        self.a2 = act1.dot(W2)

        self.probs = self.softmax(self.a2)

        return None

    def calculate_loss(self, X, t):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.ForwardPass(X, lambda x: self.actFun(x, self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        data_loss =(1/num_examples)* np.sum((-t*np.log(self.probs)))

        #data_loss =
        self.losses.append(data_loss)
        return(data_loss)


    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.ForwardPass(X, lambda x: self.actFun(x, self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backwardPass(self, X, t):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE      
        delta_scores = (self.probs - t)
        dW2 = (self.act1.T).dot(delta_scores)
        db2 = np.sum(delta_scores, axis=0, keepdims=True)
        
        # Definition: delta_2 = diff_actFun(a1) * delta_scores * W2^T
        delta_2 = np.dot(delta_scores, self.W2.T) * self.diff_actFun(self.a1, self.actFun_type)

        dW1 = np.dot(X.T, delta_2)
        db1 = np.sum(delta_2, axis=0, keepdims=True)

        return dW1, dW2, db1, db2


    def fit_model(self, X, t, epsilon, num_passes=15000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        
        # moved self.losses inside fit_model so that the list is rest 
        # with each run of fit_model with different learning rate
        self.losses = []
        
        # Gradient descent implementation
        for i in range(0, num_passes):

            # Forward propagation
            self.ForwardPass(X, lambda x: self.actFun(x, self.actFun_type))
            # Backpropagation
            dW1, dW2, db1, db2 = self.backwardPass(X, t)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

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

    means = []
    hidden_units = 100
    act = "sigmoid"
#    act = "reLU"
#    act = "tanh"
    
#    model = three_layer_NN(input_layer=2, hidden_layer=units, output_layer=2, actFun_type=act)
#    mean, loss = model.fit_model(X, t, 0.005)
#    means.append(mean)
#    model.visualize_decision_boundary(X, y)
#
#    plt.plot(model.losses)  
#    plt.title("Avg Loss per Iteration")
#    plt.xlabel("Iterations (1000s)")
#    plt.ylabel("Avg Loss")
#    plt.show() 

# Run the fit_model with varying learning size:
    model = three_layer_NN(input_layer=2, hidden_layer=hidden_units, output_layer=2, actFun_type=act)
    epsilons = [0.005, 0.001, 0.0001]    
#    epsilons = [0.005]
    
    track_loss = {}
    
    for epsilon in epsilons:
        print("Testing with epsilon =", epsilon)
        mean, loss = model.fit_model(X, t, epsilon)
        track_loss[epsilon] = loss
        model.visualize_decision_boundary(X, y)

    x = np.arange(len(loss))
    plt.plot(x, track_loss[0.005])
    plt.plot(x, track_loss[0.001])
    plt.plot(x, track_loss[0.0001])
    plt.title("Avg Loss per Iteration")
    plt.xlabel("45 values per Iteration")
    plt.ylabel("Avg Loss")
    plt.legend(['epsilon = 0.005', 'epsilon = 0.001', 'epsilon = 0.0001'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()


