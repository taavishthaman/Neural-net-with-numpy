import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
np.random.seed(5)

#Toy dataset
X, y = datasets.make_moons(n_samples = 500, noise=0.1)

y = y.reshape((len(y),1))
X = X.T 
y = y.T

#Sigmoid function
def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s

#This function returns the size of input and output layers
def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    return (n_x, n_y)

#Randomly initalizing weights and biases
def initialize_parameters(n_x, n_h, n_y):
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    param_dict = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2}
    
    return param_dict


def forward_prop(X, param_dict):
    
    W1 = param_dict["W1"]
    b1 = param_dict["b1"]
    W2 = param_dict["W2"]
    b2 = param_dict["b2"]
    
    #Forward prop
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    #Forward prop
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1" : Z1,
             "A1" : A1,
             "Z2" : Z2,
             "A2" : A2}
    
    return A2, cache


def compute_cost(A2, Y):
    
    m = Y.shape[1]
        
    cost = (-1/m)*(np.sum(np.multiply(Y, np.log(A2)) + np.multiply((1 - Y) , np.log(1 - A2))))
    cost = np.squeeze(cost)
    
    assert(isinstance(cost, float))
    
    return cost

def back_prop(param_dict, cache, X, Y):
    
    m = X.shape[1]
    W2 = param_dict["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    #Back prop
    dZ2 = A2 - Y
    dW2 = (1/m) * (np.dot(dZ2, A1.T))
    db2 = (1/m) * (np.sum(dZ2, axis=1, keepdims = True))
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1/m) * (np.dot(dZ1, X.T))
    db1 = (1/m) * (np.sum(dZ1, axis=1, keepdims=True))
    #Back prop
    
    grad_dict = {"dW1" : dW1,
                 "db1" : db1,
                 "dW2" : dW2,
                 "db2" : db2}
    
    return grad_dict

def update_params(params, grad_dict, learning_rate):
    
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    
    dW1 = grad_dict["dW1"]
    db1 = grad_dict["db1"]
    dW2 = grad_dict["dW2"]
    db2 = grad_dict["db2"]
    
    #Updates
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    #Updates
    
    param_dict = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2}
    
    return param_dict 

#Neural Net model
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):
    
    n_x= layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    
    parameters = initialize_parameters(n_x, n_h, n_y)

    
    for i in range(0, num_iterations):
        
        A2, cache = forward_prop(X, parameters)
        cost = compute_cost(A2, Y)
        grads = back_prop(parameters, cache, X, Y)
        parameters = update_params(parameters, grads, learning_rate = 0.1)
        
        if(print_cost and i%1000 == 0):
            print("Cost after iteration %i: %f" % (i, cost))
            
    return parameters

def predict(params, X):
    
    A2, cache = forward_prop(X, params)
    predictions = np.round(A2)
    
    return predictions

#Number of neurons in hidden layer
n_hidden = 3

#Running the model
parameters = nn_model(X, y, n_h = n_hidden, num_iterations=25000, print_cost=True)

#Predictions returned by the trained model
predictions = predict(parameters, X)

#Accuracy achieved on input data
accuracy = float((np.dot(y, predictions.T) + np.dot(1 - y, 1 - predictions.T)) / float(y.size) * 100)
print("accuracy with "+str(n_hidden)+" neurons in hidden layers is: " + str(accuracy))


#Plotting the decision boundary

h=0.02
x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

X = X.T
y = y.T

Z = predict(parameters, np.c_[xx.ravel(), yy.ravel()].T)

Z = Z.reshape(xx.shape)
plt.title('Decision boundary, #Neurons-->' + str(n_hidden))
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.contourf(xx, yy, Z, alpha=0.5, cmap = 'coolwarm' )
plt.axis('off')
y = y.reshape(500,)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap = 'coolwarm')
plt.show()

    
    


    
    
    
    
    
    
    
    
    
    
    
    

    
