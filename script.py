import numpy as np

class neuralNetwork:

    '''
    init function
    input_layer = number of input features
    output_layer = number of output features
    weight = weights of the model
    bias = bias of the model
    hidden_layer = number of hidden layers
    '''

    def __init__(self, input_layer, output_layer, hidden_layer): 
        self.input_layer = input_layer # number of input features
        self.output_layer = output_layer # number of output features
        self.hidden_layer = hidden_layer # number of hidden layers

        self.weight_input_hidden = np.random.randn(self.hidden_layer, self.input_layer) * 0.01 # weight for input layer
        self.weight_hidden_output = np.random.randn(self.output_layer, self.hidden_layer) * 0.01 # weight for output layer

        self.bias_hidden = np.zeros((self.hidden_layer, 1)) # bias for input layer
        self.bias_output = np.zeros((self.output_layer, 1)) # bias for output layer


    '''
    linear forward function
    A = outpot of previous layer
    W = weights
    b = bias
    Z = linear output
    '''

    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        return Z


    '''
    activation function
    Z = linear output
    '''

    def activation(self, Z, activation_type="relu"):
        if activation_type == "relu":
            return np.maximum(0, Z)
        elif activation_type == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        else:
            raise ValueError("Activation type not supported")
        

    '''
    linear activation forward function
    A_prev = output of previous layer
    W = weights
    b = bias
    activation_type = activation function
    A = output of current layer
    '''

    def linear_activation_forward(self, A_prev, W, b, activation_type="relu"):
        Z, A_prev = self.linear_forward(A_prev, W, b)
        A = self.activation(Z, activation_type)
        return A


    '''
    cost function
    Y = true label
    A = predicted label
    m = number of examples
    cost = cost of the model
    '''

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
        return cost



    def backpropagation(self, X, Y, A, learning_rate, activation_type="relu"):
        pass



    def train(self, X, Y, learning_rate, activation_type="relu", epochs=1000):
        for i in range(epochs):
            pass