import numpy as np
import pandas as pd
from scipy import optimize 
from sklearn import metrics

data = pd.read_csv('ML\\adult.data')
data = pd.DataFrame(data)

# data splitting
X= data.iloc[:,:-1]  # (32560,14)
y = data.iloc[:,-1]  # (32560,)

# removing a column
X = np.array(X)
X = np.delete(X, 2, axis = 1)
X = pd.DataFrame(X)  # (32560,13)

# one hot encoding
def one_hot(data):
    
    data = np.array(data)
    cols = 0
    rows = data.shape[0]

    try:
        for col in range(data.shape[1]):
            unique = np.unique(data[:,col])
            num_new_cols = len(unique)
            cols += num_new_cols
    
        new_shape = (rows,cols)
        new_data = np.zeros(new_shape)
        
        j = 0
        for col in range(data.shape[1]):
            unique = np.unique(data[:,col])
            for label in unique:
                new_col = np.array([1 if label == data[i,col] else 0 for i in range(rows)])
                for i in range(rows):
                    new_data[i,j]= new_col[i]
                j += 1
                print("One-Hot Encoding; Columns left: ", cols - j )
    
    except:
        for col in range(1):
            unique = np.unique(data)
            num_new_cols = len(unique)
            cols += num_new_cols
    
        new_shape = (rows,cols)
        new_data = np.zeros(new_shape)
        
        j = 0
        for col in range(1):
            unique = np.unique(data)
            for label in unique:
                new_col = np.array([1 if label == data[i] else 0 for i in range(rows)])
                for i in range(rows):
                    new_data[i,j]= new_col[i]
                j += 1
                print("One-Hot Encoding; Columns left: ", cols - j )
        
    return new_data

# one hot encoding X, y sets
X = one_hot(X)  # 32560,496
y = one_hot(y)  # 32560, 2

# adding bias 
X = np.insert(X, 0, values = np.ones(X.shape[0]), axis = 1)  # 32560,497

# determine layer sizes 
input_size = X.shape[1]  # bias already added
hidden_size1 = 20
hidden_size2 = 10
output_size = y.shape[1]

# generate random weight values 
weights = np.array(np.random.random(size = input_size*hidden_size1 + (hidden_size1+1)*hidden_size2 + 
                           (hidden_size2 +1)*output_size ) * 0.25) - 0.5
                            
# Sigmoid activation function
def g(z):
    sig = 1 / (1 + np.exp(-z))
    return sig

# gradient activation function
def grad_g(z):
    grad = np.multiply(g(z), (1 - g(z)))
    return grad

# forward propagation
def forward(weights, train_X):
    
    theta1 = weights[:input_size*hidden_size1].reshape(input_size, hidden_size1)  # 497,20
    theta2 = weights[input_size*hidden_size1 : 
        input_size*hidden_size1 + (hidden_size1+1)*hidden_size2].reshape(hidden_size1 +1, hidden_size2)  # 21,10
    theta3 = weights[input_size*hidden_size1 + (hidden_size1+1)*hidden_size2:].reshape(hidden_size2+1,output_size)  # 11,2
    
    X = np.matrix(train_X)
    theta1 = np.matrix(theta1)
    theta2 = np.matrix(theta2)
    theta3 = np.matrix(theta3)
    
    a1 = X  # bias already added 
    
    z2 = a1 * theta1
    a2 = g(z2)
    a2 = np.insert(a2, 0, values = np.ones(a2.shape[0]), axis = 1)
    
    z3 = a2 * theta2
    a3 = g(z3)
    a3 = np.insert(a3, 0, values = np.ones(a3.shape[0]), axis = 1)
    
    z4 = a3 *theta3
    a4 = g(z4)
    h = a4

    return a1, z2, a2, z3, a3, z4, h

# cost function     
def cost(weights, train_X, y, lamb):
    
    theta1 = weights[:input_size*hidden_size1].reshape(input_size, hidden_size1)  # 497,20
    theta2 = weights[input_size*hidden_size1 : 
        input_size*hidden_size1 + (hidden_size1+1)*hidden_size2].reshape(hidden_size1 +1, hidden_size2)  # 21,10
    theta3 = weights[input_size*hidden_size1 + (hidden_size1+1)*hidden_size2:].reshape(hidden_size2+1,output_size)  # 11,2
    

    a1, z2, a2, z3, a3, z4, h = forward(weights, train_X)
    
    y = np.matrix(y)  # 32560,2
    X = np.matrix(train_X) 
    theta1 = np.matrix(theta1)  # 497,20
    theta2 = np.matrix(theta2)  # 21,10
    theta3 = np.matrix(theta3)  # 11,2
    
    m = X.shape[0]
    
    J = 0
    for k in range(y.shape[1]):
        term = (-y[:,k]).T * np.log(h[:,k]) - (1 - y[:,k]).T * np.log(1 - h[:,k])
        J += (1/m) * float(term)
        
    term2 = float(theta1[:,1:].flatten() * theta1[:,1:].flatten().T)
    term3 = float(theta2[:,1:].flatten() * theta2[:,1:].flatten().T)
    term4 = float(theta3[:,1:].flatten() * theta3[:,1:].flatten().T)
    
    J += (lamb / (2*m)) * (term2 + term3 + term4)

    return J

# back forward 
def back(weights, train_X, y, lamb):
    
    theta1 = weights[:input_size*hidden_size1].reshape(input_size, hidden_size1)  # 497,20
    theta2 = weights[input_size*hidden_size1 : 
        input_size*hidden_size1 + (hidden_size1+1)*hidden_size2].reshape(hidden_size1 +1, hidden_size2)  # 21,10
    theta3 = weights[input_size*hidden_size1 + (hidden_size1+1)*hidden_size2:].reshape(hidden_size2+1,output_size)  # 11,2
    
    a1, z2, a2, z3, a3, z4, h = forward(weights, train_X)
    
    y = np.matrix(y)  # 32560,2
    X = np.matrix(train_X) 
    theta1 = np.matrix(theta1)  # 497,20
    theta2 = np.matrix(theta2)  # 21,10
    theta3 = np.matrix(theta3)  # 11,2
    
    m = X.shape[0]
    
    d4 = h - y  # 32560,2
    
    d3 = d4 * theta3.T
    d3 = np.multiply(d3[:,1:], grad_g(z3))
    
    d2 = d3 * theta2.T
    d2 = np.multiply(d2[:,1:], grad_g(z2))
    
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    delta3 = np.zeros(theta3.shape)
    
    grad1 = np.zeros(theta1.shape)
    grad2 = np.zeros(theta2.shape)
    grad3 = np.zeros(theta3.shape)
    
    delta1 = delta1 + a1.T * d2
    delta2 = delta2 + a2.T * d3
    delta3 = delta3 + a3.T * d4
    
    grad1[:,1:] = (1/m) * delta1[:,1:] + (lamb/m) * theta1[:,1:]
    grad2[:,1:] = (1/m) * delta2[:,1:] + (lamb/m) * theta2[:,1:]
    grad3[:,1:] = (1/m) * delta3[:,1:] + (lamb/m) * theta3[:,1:]
    
    grad1 = grad1.ravel()
    grad2 = grad2.ravel()
    grad3 = grad3.ravel()
     
    grad = np.concatenate((np.concatenate((grad1,grad2),axis = 0), grad3), axis=0)

    return grad

# training nn model and predicting
def train(weights, X, y, epochs, lamb):
    
    """
    weights: the original randomly generated weights
    epochs: chosen number times 10
    cost function is displayed after a set of 10 epochs 
    """
    
    for i in range(epochs):
        print("Cost" + str(i + 1),":", cost(weights, X, y, lamb))
        weights = optimize.minimize(cost, x0 = weights, args = (X, y, lamb) ,method = 'TNC', jac = back, options = {'maxiter':10})
        weights = weights.x
        
    print("Final cost function value: ", cost(weights, X, y, lamb)) 

    return weights

# predictions
def prediction(weights, X):
    a1, z2, a2, z3, a3, z4, h = forward(weights, X)
    
    argmax = np.argmax(h, axis = 1)

    predicted_y = np.zeros((argmax.shape[0],2))
    
    rows = argmax.shape[0]
    for i in range(rows):
        if argmax[i] == 0:
            predicted_y[i,0] = 1
            predicted_y[i,1] = 0
        else:
            predicted_y[i,1] = 1
            predicted_y[i,0] = 0
    
    return predicted_y

new_weights = train(weights, X, y, 11, 1)

# get the predictions
predicted_y = prediction(new_weights, X)

# prediction accuracy
accuracy = metrics.accuracy_score(y,predicted_y)
print(accuracy*100)

pred = prediction(new_weights, X[3000, :])
print(pred)
