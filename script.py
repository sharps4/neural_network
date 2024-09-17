import numpy as np

weight = np.random.rand(4, 3)
bias = np.random.rand(4, 1)


'''
linear forward function
A = outpot of previous layer
W = weights
b = bias
Z = linear output
'''

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    return Z


'''
activation function
Z = linear output
'''

def activation(Z, activation_type="relu"):
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

def linear_activation_forward(A_prev, W, b, activation_type="relu"):
    Z, A_prev = linear_forward(A_prev, W, b)
    A = activation(Z, activation_type)
    return A


'''
cost function
Y = true label
A = predicted label
m = number of examples
cost = cost of the model
'''

def cost(Y, A):
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    return cost

