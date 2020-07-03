import numpy as np
import pandas as pd

names = ['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','Class']

data = pd.read_csv('ML\\poker-hand-training-true.data', names = names)
data = np.array(data)
cols = data.shape[1]

# Splitting data
X = data[:, :cols-1]
y = data[:, cols-1:cols]

# Preprocessing data
def preprocess_X(X):
    array = np.ones((X.shape[0],1))
    for j in range(X.shape[1]):
        unique = np.unique(X[:,j]).shape[0]
        for i in range(1, unique + 1):
            new_column = np.array([1 if label == i else 0 for label in X[:,j]])
            array = np.column_stack((array,new_column))
    return array

def preprocess_y(y):
    unique_y = np.unique(y).shape[0]
    array = np.ones((y.shape[0],1))
    for k in range(unique_y):
        new_column = np.array([1 if label == k else 0 for label in y])
        array = np.column_stack((array,new_column))
    train_y = array[:,1:]
    return train_y

train_X = preprocess_X(X)
train_y = preprocess_y(y)


# the size of input, hidden, and output layers
input_size = train_X.shape[1]  # here one bias node is already added
hidden_size1 = 50
hidden_size2 = 20
output_size = train_y.shape[1]

# generate initial weights
weights = (np.random.random(size = input_size * hidden_size1 + ( hidden_size1 + 1 )*hidden_size2 
                           + (hidden_size2 + 1)*output_size ) - 0.5) * 0.25 

lamb = 1

# activation function
def act(z): 
    return 1 / (1 + np.exp(-z))

# gradient of activation function
def der_act(z):
    sig = act(z)
    return np.multiply(sig, (1-sig))

# forward propagation
def forward(weights, train_X):
    
    theta1 = np.reshape(weights[:input_size*hidden_size1], (input_size, hidden_size1))

    theta2 = np.reshape(weights[input_size * hidden_size1:input_size * hidden_size1 + ( hidden_size1 + 1 )*hidden_size2 ]
            , (hidden_size1 + 1, hidden_size2))

    theta3 = np.reshape(weights[input_size * hidden_size1 + ( hidden_size1 + 1 )*hidden_size2:
    input_size * hidden_size1 + ( hidden_size1 + 1 )*hidden_size2 + (hidden_size2 + 1)*output_size],
    (hidden_size2 + 1, output_size))
    
    theta1 = np.matrix(theta1)
    theta2 = np.matrix(theta2)
    theta3 = np.matrix(theta3)
    
    a1 = train_X  # bias node already added
    z2 = a1 * theta1
    
    a2 = act(z2)
    a2 = np.insert(a2, 0, values = np.ones(train_X.shape[0]), axis = 1 )  # bias node
    z3 = a2 * theta2
    
    a3 = act(z3)
    a3 = np.insert(a3, 0, values = np.ones(train_X.shape[0]), axis = 1 )  # bias node
    
    z4 = a3 * theta3
    a4 = act(z4)
    h = a4
    
    return a1, a2, a3, h, z2, z3, z4

# cost function
def cost(weights, train_X, train_y, lamb):
    
    theta1 = np.reshape(weights[:input_size*hidden_size1], (input_size, hidden_size1))

    theta2 = np.reshape(weights[input_size * hidden_size1:input_size * hidden_size1 + ( hidden_size1 + 1 )*hidden_size2 ]
            , (hidden_size1 + 1, hidden_size2))

    theta3 = np.reshape(weights[input_size * hidden_size1 + ( hidden_size1 + 1 )*hidden_size2:
    input_size * hidden_size1 + ( hidden_size1 + 1 )*hidden_size2 + (hidden_size2 + 1)*output_size],
    (hidden_size2 + 1, output_size))
    
    theta1 = np.matrix(theta1)
    theta2 = np.matrix(theta2)
    theta3 = np.matrix(theta3)
    
    a1, a2, a3, h, z2, z3, z4 = forward(weights, train_X)
    K = train_y.shape[1]
    m = train_X.shape[0]
    J = 0
    
    # loss term
    for i in range(K):
        term = - train_y[:,i].T * np.log(h[:,i]) - (1 - train_y[:,i]).T * np.log(1-h[:,i])
        J += term / m
    
    # regularization terms
    theta1 = theta1[1:,:]
    theta2 = theta2[1:,:]
    theta3 = theta3[1:,:]
    K1 = theta1.shape[1]
    K2 = theta2.shape[1]
    K3 = theta3.shape[1]
    
    for j in range(K1):
        term1 = theta1[:,j].T * theta1[:,j]
        J += (lamb / (2 * m)) * term1
    
    for k in range(K2):
        term2 = theta2[:,k].T * theta2[:,k]
        J += (lamb / (2 * m)) * term2
    
    for l in range(K3):
        term3 = theta3[:,l].T * theta3[:,l]
        J += (lamb / (2 * m)) * term3
        
    return float(J)

# backward propagation
def backward(weights, train_X, train_y, lamb):
    
    theta1 = np.reshape(weights[:input_size*hidden_size1], (input_size, hidden_size1))

    theta2 = np.reshape(weights[input_size * hidden_size1:input_size * hidden_size1 + ( hidden_size1 + 1 )*hidden_size2 ]
            , (hidden_size1 + 1, hidden_size2))

    theta3 = np.reshape(weights[input_size * hidden_size1 + ( hidden_size1 + 1 )*hidden_size2:
    input_size * hidden_size1 + ( hidden_size1 + 1 )*hidden_size2 + (hidden_size2 + 1)*output_size],
    (hidden_size2 + 1, output_size))
    
    m = train_X.shape[0]
    
    a1, a2, a3, h, z2, z3, z4 = forward(weights, train_X)
    theta1 = np.matrix(theta1)
    theta2 = np.matrix(theta2)
    theta3 = np.matrix(theta3)
    
    d4 = h - train_y
    
    d3 = (d4 * theta3.T)[:,1:]
    d3 = np.multiply(d3, der_act(z3) )
    
    d2 = (d3 * theta2.T)[:,1:]
    d2 = np.multiply(d2, der_act(z2))
    
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    delta3 = np.zeros(theta3.shape)
    
    delta1 = delta1 + a1.T*d2
    delta2 = delta2 + a2.T*d3
    delta3 = delta3 + a3.T*d4
    
    grad1 = np.zeros(delta1.shape)
    grad1[1:,:] = delta1[1:,:] / m + (lamb / m) * theta1[1:,:]
    
    grad2 = np.zeros(delta2.shape)
    grad2[1:,:] = delta2[1:,:] / m + (lamb / m) * theta2[1:,:]
    
    grad3 = np.zeros(delta3.shape)
    grad3[1:,:] = delta3[1:,:] / m + (lamb / m) * theta3[1:,:]
    
    grad1 = grad1.ravel()
    grad2 = grad2.ravel()
    grad3 = grad3.ravel()
    
    grad = np.concatenate( (np.concatenate((grad1,grad2), axis = 0), grad3), axis = 0)
    
    return grad

# optimization
from scipy import optimize 
from sklearn import metrics

e = 20  # number of epoch sets

# training the model
cost1 = cost(weights, train_X, train_y, lamb)
print('1)', cost1)
fmin = optimize.minimize(fun = cost, x0 = weights, args = (train_X, train_y, lamb), method = 'TNC', jac = backward, options = {'maxiter': 50})

new_weights = fmin.x
cost2 = cost(new_weights, train_X, train_y, lamb)
print('2)', cost2)
        
for i in range(3, e + 1):
    fmin = optimize.minimize(fun = cost, x0 = new_weights, args = (train_X, train_y, lamb), method = 'TNC', jac = backward, options = {'maxiter': 50})
    new_weights = fmin.x
    cost3 = cost(new_weights, train_X, train_y, lamb)
    print(i,')',cost3)

# prediction accuracy 
a1, a2, a3, h, z2, z3, z4 = forward(new_weights, train_X)
new_y = h

pred = np.argmax(new_y, axis= 1)
accuracy = metrics.accuracy_score(y,pred)
    
print(accuracy * 100, '%')
