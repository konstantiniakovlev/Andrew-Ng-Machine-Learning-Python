import numpy as np
import pandas as pd 
from scipy.io import loadmat
from scipy import optimize, signal


"DATA"
data = loadmat('ML\\ex3data1.mat')

X = data['X']
y = data['y']

def y_onehot(y):
    col = len(np.unique(y))
    rows = y.shape[0]
    y_all = np.zeros((rows, col))
    for i in range(1, col + 1):
        y_i = [1 if label == i else 0 for label in y]
        for j in range(rows):
            y_all[j,i-1] = y_i[j]
    
    return y_all

y_oh = y_onehot(y) # 5000,10

input_size = 400
hidden_size = 25
output_size = 10


weights = (np.random.random(size = hidden_size * (input_size +1) + output_size *(hidden_size +1)  )-0.5)*0.25 


theta1 = np.reshape(weights[:hidden_size * (input_size +1)], ( hidden_size, input_size+1 ) ) # 25,401
theta2 = np.reshape(weights[hidden_size * (input_size +1):], ( output_size, hidden_size+1, ) ) # 10,26

#------------------------------------------------------------------------------

def g(z):
    return 1 / (1 + np.exp(-z))

"Sigmoid Gradient"
def sig_grad(z):
    sig = g(z)
    return np.multiply(sig, (1-sig))

"Forward Propagation"
def forwardfeed(X, theta1, theta2):
    X =np.matrix(X)
    theta1 = np.matrix(theta1)
    theta2 = np.matrix(theta2)
    'step1'
    a1 = X
    rows = X.shape[0]
    a1 = np.insert(a1, 0, values = np.ones(rows), axis = 1) #5000,401
    'step2'
    z2 = (theta1 * a1.T ).T # 5000, 25
    a2 = g(z2) # 5000, 25
    a2 = np.insert(a2, 0, values = np.ones(rows), axis = 1) #5000,26
    'step3'
    z3 = (theta2 * a2.T ).T
    a3 = g(z3)
    h = a3 # 5000,10
    
    return a1, a2, z2, z3, h

"Cost Function"
def cost(param, X, y_oh, lamb):
    X = np.matrix(X)
    y_oh=np.matrix(y_oh)
    
    
    theta1 = np.matrix(np.reshape(param[:hidden_size * (input_size +1)], ( hidden_size, input_size+1 ) ))# 25,401
    theta2 = np.matrix(np.reshape(param[hidden_size * (input_size +1):], ( output_size, hidden_size+1, ) )) # 10,26


    
    a1, a2, z2, z3, h = forwardfeed(X, theta1, theta2)
    J = 0
    m = y_oh.shape[0]
    K = y_oh.shape[1]
    for k in range(K):
        term1 = -y_oh[:,k].T * np.log(h[:,k])
        term2 = -(1 - y_oh[:,k]).T * np.log(1 - h[:,k])
        J += (term1+term2)/m
        
    theta1 = np.matrix(theta1[:,1:])
    theta2 = np.matrix(theta2[:,1:])
    
    K = theta1.shape[1]
    for k in range(K):
        term3 = theta1[:,k].T * theta1[:,k]
        term3 = term3 * (lamb / (2*m))
        J += term3
        
    L = theta2.shape[1]
    for l in range(L):
        term4 = theta1[:,l].T * theta1[:,l]
        term4 = term4 * (lamb / (2*m))
        J += term4
    
    return float(J)



"Backward Propagation"
def backwardfeed(param, X, y_oh, lamb):
    X = np.matrix(X)
    y_oh=np.matrix(y_oh)
    m = X.shape[0]
    
    theta1 = np.reshape(param[:hidden_size * (input_size +1)], ( hidden_size, input_size+1 ) ) # 25,401
    theta2 = np.reshape(param[hidden_size * (input_size +1):], ( output_size, hidden_size+1, ) ) # 10,26
    
    theta1 = np.matrix(theta1)
    theta2 = np.matrix(theta2)
    
    a1, a2, z2, z3, h = forwardfeed(X, theta1, theta2)
    
    d3 = h - y_oh #5000,10
    gz2 = sig_grad(z2) #5000, 25
    d2_1 = (d3 * theta2)[:,1:]
    d2 = np.multiply(d2_1,gz2) # 5000,25
    
    delta1 = np.zeros(theta1.shape) #25,401
    delta2 = np.zeros(theta2.shape) #10, 26
    
    delta1 = delta1 + d2.T * a1 #25,401 + 25,5000 * 5000,401
    delta2 = delta2 + d3.T * a2 #10,26 + 10,5000 * 5000,26
    
    "Not REGULARIZING THE FIRST TERM"
    delta1[:,1:] = delta1[:,1:] / m + theta1[:,1:] * (lamb/(2*m))
    delta2[:,1:] = delta2[:,1:] / m + theta2[:,1:] * (lamb/(2*m))
    
    grad = np.concatenate((delta1.ravel(), delta2.ravel()),axis =1 )
    return grad

lamb = 1

c1 = cost(weights, X, y_oh, lamb)    
print(0,') cost: ', c1)

fmin = optimize.minimize(fun = cost, x0 = weights, args=(X,y_oh,lamb), method = 'TNC', jac = backwardfeed)
c2 = cost(fmin.x, X, y_oh, lamb) 
print(1,') cost: ',c2) 
for i in range(9):
    fmin = optimize.minimize(fun = cost, x0 = fmin.x, args=(X,y_oh,lamb), method = 'TNC', jac = backwardfeed)
    print(i+2,') cost: ', cost(fmin.x, X, y_oh, lamb))

new_theta1 = np.reshape(fmin.x[:hidden_size * (input_size +1)], ( hidden_size, (input_size+1) ) ) # 25,401
new_theta2 = np.reshape(fmin.x[hidden_size * (input_size +1):], ( output_size, (hidden_size+1) ) ) # 10,26


a1, a2, z2, z3, h = forwardfeed(X, new_theta1, new_theta2)

y_pred = np.array(np.argmax(h, axis=1) + 1)

def accuracy_score(y,y_pred):
    ac_list = []
    rows = y.shape[0]
    for i in range(rows):
        if y[i] == y_pred[i]:
            ac_list.append(1)
    accuracy = np.sum(ac_list) / len(y)
    return accuracy

accuracy = accuracy_score(y,y_pred)

        
print("Accuracy Score:", accuracy*100, "%")








