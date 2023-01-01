__author__ = 'Yuan Yuan'

from A1.three_layer_neural_network import NeuralNetwork
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    # X, y = datasets.make_moons(200, noise=0.20)
    # X, y = datasets.make_circles(200,noise = 0.20)
    X, y = datasets.make_blobs(200)
    return X, y

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
########################################################################################################################
# MY ASSSIGMENT STARTS HERE
########################################################################################################################
########################################################################################################################

class DeepNeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_input_dim, nn_layers, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_layers = nn_layers
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.WO = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.bO = np.zeros((1, self.nn_output_dim))

        # initialize hidden layers
        self.hiddenLayers = []
        for i in range(self.nn_layers):
            if i == 0:
                self.hiddenLayers.append(Layer(i, nn_input_dim, nn_hidden_dim, actFun_type, reg_lambda, seed))
            elif i == self.nn_layers-1:
                self.hiddenLayers.append(Layer(i, nn_hidden_dim, nn_output_dim, actFun_type, reg_lambda, seed))
            else:
                self.hiddenLayers.append(Layer(i, nn_hidden_dim, nn_hidden_dim, actFun_type, reg_lambda, seed))

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE

        self.z1 = np.dot(X,self.W1) + self.b1
        self.a1 = actFun(self.z1)
        #print('feedforword X = {}'.format(X.shape))
        self.hiddenLayers[0].feedforward(X)
        for i in range(1,self.nn_layers):
            self.hiddenLayers[i].feedforward(self.hiddenLayers[i-1].layer_a)
            #print('layer_a is {},layer_W is {},layer_b is {}'.format(str(self.hiddenLayers[i].layer_a.shape),
            #                                                         str(self.hiddenLayers[i].layer_W.shape),
            #                                                         str(self.hiddenLayers[i].layer_b.shape)))
        self.zO = self.hiddenLayers[-1].layer_z
        exp_sum = np.exp(self.zO - np.max(self.zO))
        self.probs =exp_sum/np.sum(exp_sum, axis = 1, keepdims = True)

        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: NeuralNetwork.actFun(x, type=self.actFun_type))
        # Calculating the loss

        data_loss = np.sum(-np.log(self.probs[range(num_examples), y]))

        # Add regulatization term to loss (optional)
        regSum = sum([np.sum(np.square(layer.layer_W)) for layer in self.hiddenLayers])
        regSum += np.sum(np.square(self.W1)) + np.sum(np.square(self.WO))
        data_loss += self.reg_lambda / 2 * regSum
        data_loss = (1. / num_examples) * data_loss
        return data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: NeuralNetwork.actFun(x, type=self.actFun_type))

        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE

        dldz2 = self.probs
        dldz2[range(len(X)),y] -= 1 #dldzO
        dldz2 /= len(X)
        # dldz1 = dlda1 * da1dz1

        self.hiddenLayers[-1].layer_delta = dldz2
        self.hiddenLayers[-1].layer_dw = np.dot(self.hiddenLayers[-1].layerInput.T, dldz2) #dWO
        self.hiddenLayers[-1].layer_db = np.sum(dldz2, axis = 0, keepdims = True) #dbO

        for i in range(self.nn_layers-2, -1, -1):
            self.hiddenLayers[i].backprop(self.hiddenLayers[i+1].layer_W, self.hiddenLayers[i+1].layer_delta)

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: NeuralNetwork.actFun(x, type=self.actFun_type))
            # Backpropagation
            #dW1, dW2, db1, db2 = self.backprop(X, y)
            self.backprop(X, y)

            # update weight and bias for the hidden layers
            for layer in self.hiddenLayers:
                layer.layer_dw +=self.reg_lambda * layer.layer_W
                layer.layer_W += -epsilon * layer.layer_dw
                layer.layer_b += -epsilon * layer.layer_db

            #dW1 += self.reg_lambda * self.W1
            #self.W1 += -epsilon * dW1
            #self.b1 += -epsilon * db1
            self.W1 = self.hiddenLayers[0].layer_W
            self.b1 = self.hiddenLayers[0].layer_b
            self.WO = self.hiddenLayers[-1].layer_W
            self.bO = self.hiddenLayers[-1].layer_b
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


class Layer(object):
    def __init__(self,layerIndex,dim0,dim1,actFun_type,reg_lambda,seed):

        self.dim0 = dim0
        self.dim1 = dim1
        self.layerIndex = layerIndex
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize W & b for each layer
        np.random.seed(seed)
        self.layer_W = np.random.randn(self.dim0, self.dim1) / np.sqrt(self.dim0)
        self.layer_b = np.zeros((1, self.dim1))

    def feedforward(self, input):

        self.layerInput = input
        self.layer_z = np.dot(input,self.layer_W) + self.layer_b
        #print('ID = {},input = {}, dot = {}, layer_b = {}'.format(str(self.layerIndex),str(self.layerInput.shape),str(np.dot(input,self.layer_W).shape), str(self.layer_b.shape)))
        self.layer_a = NeuralNetwork.actFun(self.layer_z, self.actFun_type)

    def backprop(self, W, lastDelta):
        #print('W is {},ID = {},lastDelta is {}'.format(str(W.shape),str(self.layerIndex),str(lastDelta.shape)))
        #print(self.diff_actFun(self.layer_z,self.actFun_type))
        #print('layerz is {}'.format(self.layer_z.shape))
        #print('layer_input is {}'.format(self.layerInput.shape))
        self.layer_delta = np.dot(lastDelta, np.transpose(W)) * NeuralNetwork.diff_actFun(self.layer_z,self.actFun_type)
        self.layer_dw = np.dot(self.layerInput.T,self.layer_delta)
        self.layer_db = np.sum(self.layer_delta,axis = 0,keepdims = True)


def main():
    # # generate and visualize Make-Moons dataset
     X, y = generate_data()
     #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
     #plt.show()

     model = DeepNeuralNetwork(nn_input_dim=2, nn_layers=7, nn_hidden_dim = 7, nn_output_dim=3, actFun_type='tanh')
     model.fit_model(X, y)
     model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()
