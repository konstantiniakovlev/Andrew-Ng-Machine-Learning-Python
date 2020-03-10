import numpy as np 
import pandas as pd
import matplotlib.pyplot 
from scipy.io import loadmat
from scipy import optimize 
from sklearn import metrics

data = loadmat('ML\\ex3data1.mat',)

X = data['X']
X = np.insert(X, 0, values = np.ones(X.shape[0]), axis = 1 ) #(5000,401)
y = data['y'] #(5000,1)


def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

lamb = 1
parameters = X.shape[1]
theta = np.zeros(parameters)

def LossFunction(theta, X, y, lamb): #scalar
    m = len(X)
    theta = np.matrix(theta)#(1,5000)
    X = np.matrix(X)
    y = np.matrix(y)
    
    "loss function"
    sigmoid = Sigmoid(X * theta.T) #(5000,1)  
    loss_term = y.T * np.log(sigmoid) + (1-y).T * np.log(1-sigmoid)
    reg = lamb * (theta * (theta.T))
    loss = - ( loss_term / m ) + (reg / (2*m))
    return loss

print("Loss before optimization: ", int(LossFunction(theta,X,y,lamb)))

def Gradient(theta, X, y, lamb):
    m = len(X)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    sigmoid = Sigmoid(X * theta.T)
    error = sigmoid - y
    term1 = error.T * X
    term2 = 2 * lamb * theta
    
    grad = term1 / m + term2 / (2*m)
    return grad

number_of_labels = len(np.unique(y)) #10
parameters = int(X.shape[1])
all_theta = np.zeros((number_of_labels , parameters))
    
def oneVSall(X, y, lamb):
    number_of_labels = len(np.unique(y)) #10
    parameters = int(X.shape[1])
    all_theta = np.zeros((number_of_labels , parameters))
    
    for i in range(1, number_of_labels + 1):
        theta = np.zeros(parameters)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.matrix(y_i).T
        print(i)
        theta_i = optimize.minimize(fun = LossFunction, x0 = theta, args = (X, y_i, lamb), method = 'TNC', jac = Gradient)
        for j in range(parameters):
            all_theta[i-1,j] = theta_i['x'][j] #theta_i['x'] (401,)
            
    return all_theta
    
full_theta = oneVSall(X, y, lamb) #(10,401)

def predictions(X, theta):
    X = np.matrix(X)
    theta = np.matrix(theta)
    z = X * theta.T
    h = Sigmoid(z) # (5000, 10)
    
    best = np.argmax(h, axis = 1) + 1 #(5000,1)
    
    return best
    
prediction = predictions(X, full_theta)

def AccuracyScore(y, prediction):
    score_list = []
    for i in range(y.shape[0]):
        if prediction[i] == y[i]:
            score_list.append(1)
        else:
            score_list.append(0)
            
    score_list = np.array(score_list)
    accuracy_score = np.sum(score_list) / len(score_list)
    return accuracy_score

score = AccuracyScore(y, prediction) * 100
print("Regularization parameter: ", lamb, "\nAccuracy score: ", score,"%\nOptimization model: TNC")
    


    
    
    
    
    


