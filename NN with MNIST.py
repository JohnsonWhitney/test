#The code for the Neural Network

# Package imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.datasets import make_classification

def sigmoid(z):
    """
    Compute the sigmoid of z, a commonly used activation function for neural computation

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1/(1+np.exp(-z))

    return s

# GRADED FUNCTION: layer_sizes

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer
 
    return (n_x, n_h, n_y)



def initialize_network(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your network:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(n_h, n_x) #* .01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) #* .01
    b2 = np.zeros((n_y, 1))
    ### END CODE HERE ###
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    network = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return network

def forward_propagation(X, network):
    """
    Argument:
    X -- input data of size (n_x, m)
    network -- python dictionary containing your network (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    activations -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "network"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = network["W1"]
    b1 = network["b1"]
    W2 = network["W2"]
    b2 = network["b2"]
    ### END CODE HERE ###
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    linA1 = np.dot(W1,X) + b1
    A1 = np.tanh(linA1)
    linA2 = np.dot(W2,A1) + b2
    A2 = sigmoid(linA2)

    
    assert(A2.shape == (10, X.shape[1]))
    
    activations = {"linA1": linA1,
             "A1": A1,
             "linA2": linA2,
             "A2": A2}
    
    return A2, activations

def evaluate(A2, Y):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    network -- python dictionary containing your network W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """

    # Compute the cross-entropy cost

    logprobs = np.multiply(np.log(A2),Y)
    cost = - np.sum(logprobs)

    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(network, activations, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    network -- python dictionary containing our network 
    activations -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    gradients -- python dictionary containing your gradients with respect to different network
    """
    # First, retrieve W2 and W1 from the dictionary "network".
    W2 = network["W2"]
    W1 = network["W1"]          #not actually used

    # Now retrieve A2 and A1 from the dictionary "activations"        
    A2 = activations["A2"]
    A1 = activations["A1"]

    
    #we don't want the number of samples to drastically effect true learning rate
    #so we're going to normalize for that by dividing by that number
    n_samples = X.shape[1]
    
    # Backward propagation: calculate ddW1, ddb1, ddW2, ddb2. 
    
    ddlinA2 = A2 - Y
    ddW2 = 1/n_samples*np.dot(ddlinA2,A1.T)
    ddb2 = 1/n_samples*np.sum(ddlinA2, axis = 1, keepdims = True)
    
    ddlinA1 = np.dot(W2.T,ddlinA2)*(1 - np.power(A1,2))
    ddW1 = 1/n_samples*np.dot(ddlinA1,X.T)
    ddb1 = 1/n_samples*np.sum(ddlinA1, axis = 1, keepdims = True)

    
    gradients = {"ddW1": ddW1,
             "ddb1": ddb1,
             "ddW2": ddW2,
             "ddb2": ddb2}
    
    return gradients

def learn(network, gradients, learning_rate = 1.2):
    """
    Updates network using the gradient descent update rule given above
    
    Arguments:
    network -- python dictionary containing your network 
    gradients -- python dictionary containing your gradients 
    
    Returns:
    network -- python dictionary containing your updated network 
    """
    # Retrieve each parameter from the dictionary "network"
    W1 = network["W1"]
    b1 = network["b1"]
    W2 = network["W2"]
    b2 = network["b2"]
    
    # Retrieve each gradient from the dictionary "gradients"
    ddW1 = gradients["ddW1"]
    ddb1 = gradients["ddb1"]
    ddW2 = gradients["ddW2"]
    ddb2 = gradients["ddb2"]
    
    # Update rule for each parameter
    W1 = W1 - learning_rate*ddW1
    b1 = b1 - learning_rate*ddb1
    W2 = W2 - learning_rate*ddW2
    b2 = b2 - learning_rate*ddb2
    
    network = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return network

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    network -- network learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(4)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    costs = []
    
    # Initialize network, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, network".
    network = initialize_network(n_x, n_h, n_y)
    W1 = network["W1"]
    b1 = network["b1"]
    W2 = network["W2"]
    b2 = network["b2"]
    
    plt.figure()
    # Loop (gradient descent)
    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, network". Outputs: "A2, activations".
        A2, activations = forward_propagation(X, network)
        
        # Cost function. Inputs: "A2, Y, network". Outputs: "cost".
        cost = evaluate(A2, Y)
 
        # Backpropagation. Inputs: "network, activations, X, Y". Outputs: "gradients".
        gradients = backward_propagation(network, activations, X, Y)
 
        # Gradient descent parameter update. Inputs: "network, gradients". Outputs: "network".
        network = learn(network, gradients, learning_rate = 1.2)
        
        # Print the cost every 1000 iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(1.2)) #learning_rate))
    plt.show()
    plt.clf()
    return network

################################################################
#################33#############################################
################################################################
    
#This is code for getting MNIST data
import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training data shape: ", x_train.shape) # (60000, 28, 28) -- 60000 images, each 28x28 pixels
print("Test data shape", x_test.shape) # (10000, 28, 28) -- 10000 images, each 28x28

# Flatten the images
image_vector_size = 28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

print("Training label shape: ", y_train.shape) # (60000,) -- 60000 numbers (all 0-9)
print("First 5 training labels: ", y_train[:5]) # [5, 0, 4, 1, 9]

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("First 5 training lables as one-hot encoded vectors:\n", y_train[:5])

# This is the one-hot version of: [5, 0, 4, 1, 9]
"""
[[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
"""

#to run our model using MNIST Data:
part_x_train = x_train [0:100,:]
part_y_train = y_train [0:100,:]
network = nn_model(part_x_train.T, part_y_train.T, n_h = 20, num_iterations = 2000, print_cost=True)

def predict(network, X):
    """
    Using the learned network, predicts a class for each example in X
    
    Arguments:
    network -- python dictionary containing your network 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    A2, activations = forward_propagation(X, network)
    predictions = np.argmax(A2, axis=0)
    predictions_onehot = keras.utils.to_categorical(predictions, 10).T

    
    return predictions, predictions_onehot

[train_predictions, train_predictions_onehot] = predict(network, part_x_train.T)
[test_predictions, test_predictions_onehot] = predict(network, x_test.T)

def grade(predictions,Y):
    deviations = np.absolute(predictions - Y)
    errors = np.sum(np.amax(deviations, axis=0))
    n_samples = predictions.shape[1]
    score = (n_samples - errors)/n_samples
    
    return score

train_score = grade(train_predictions_onehot, part_y_train.T)
print('Train Accuracy: %d' %float(100*train_score) + '%')
                    
test_score = grade(test_predictions_onehot, y_test.T)
print('Test Accuracy: %d' %float(100*test_score) + '%')

sample = 79
img = x_train[sample]
img = img.reshape((28,28))
plt.imshow(img)
print('We predicted ' + str(train_predictions[sample]))