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
        Z = self.linear_forward(A_prev, W, b)
        A = self.activation(Z, activation_type)
        return A, Z


    '''
    cost function
    Y = true label
    A = predicted label
    m = number of examples
    epsilon = to avoid division by zero
    cost = cost of the model
    '''

    def cost(self, Y, A):
        m = Y.shape[1]
        epsilon = 1e-8  # to avoid division by zero
        cost = -1/m * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
        return cost


    '''
    relu derivative function
    Z = linear output
    '''

    def relu_derivative(self, Z):
        return Z > 0
    

    '''
    backpropagation function
    X = input features
    Y = true label
    cache = intermediate values
    learning_rate = learning rate
    activation_type = activation function
    '''

    def backpropagation(self, X, Y, cache, learning_rate):
        m = X.shape[1]
        A_last = cache["A_last"] # output of last layer
        Z_last = cache["Z_last"] # linear output of last layer
        A_prev_hidden = cache["A_hidden"] # output of hidden layer
        Z_hidden = cache["Z_hidden"] # linear output of hidden layer

        dZ_last = A_last - Y # derivative of cost with respect to Z of last layer

        dW_last = (1/m) * np.dot(dZ_last, A_prev_hidden.T) # derivative of cost with respect to weight of last layer
        db_last = (1/m) * np.sum(dZ_last, axis=1, keepdims=True) # derivative of cost with respect to bias of last layer

        dZ_hidden = np.dot(self.weight_hidden_output.T, dZ_last) * self.relu_derivative(Z_hidden) # derivative of cost with respect to Z of hidden layer
        dW_hidden = (1/m) * np.dot(dZ_hidden, X.T) # derivative of cost with respect to weight of hidden layer
        db_hidden = (1/m) * np.sum(dZ_hidden, axis=1, keepdims=True) # derivative of cost with respect to bias of hidden layer

        self.weight_hidden_output -= learning_rate * dW_last # update weight of last layer
        self.bias_output -= learning_rate * db_last # update bias of last layer
        self.weight_input_hidden -= learning_rate * dW_hidden # update weight of hidden layer
        self.bias_hidden -= learning_rate * db_hidden # update bias of hidden layer


    '''
    train function
    X = input features
    Y = true label
    learning_rate = learning rate
    epochs = number of iterations
    '''

    def train(self, X, Y, learning_rate=0.01, epochs=1000):
        for i in range(epochs):
            #forward propagation
            A_hidden, Z_hidden = self.linear_activation_forward(X, self.weight_input_hidden, self.bias_hidden, activation_type="relu")
            A_last, Z_last = self.linear_activation_forward(A_hidden, self.weight_hidden_output, self.bias_output, activation_type="sigmoid")

            #store intermediate values
            cache = {
                'A_hidden': A_hidden,
                'Z_hidden': Z_hidden,
                'A_last': A_last,
                'Z_last': Z_last
            }

            #cost
            cost = self.cost(Y, A_last)

            #backpropagation
            self.backpropagation(X, Y, cache, learning_rate)

            if i % 100 == 0:
                print(f"Coût après l'itération {i}: {cost}")


    '''
    predict function
    X = input features
    '''

    def predict(self, X):
        A_hidden, _ = self.linear_activation_forward(X, self.weight_input_hidden, self.bias_hidden, activation_type="relu")
        A_last, _ = self.linear_activation_forward(A_hidden, self.weight_hidden_output, self.bias_output, activation_type="sigmoid")
        predictions = (A_last > 0.5).astype(int)  # thresholding
        return predictions