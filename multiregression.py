import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocess import preprocess
from sklearn.metrics import r2_score

def gradient_descent(X, y, learning_rate, alpha, max_iter):
    threshold = 0.0001
    m = np.zeros(len(X[0]))
    b = np.random.randn(1)
    fx_old = np.float64('inf')
    for iteration in range(max_iter):
        fx_new = 0
        for i in range(len(X)):
            index = np.random.randint(len(X))
            x = X[index]
            y_i = y[index]
            gradient = np.float64(-2 * x * (y_i - (m.T.dot(x) + b)) + 2*alpha*m)
            m = np.float64(m - learning_rate/len(x) * gradient)
            b = np.float64(b - learning_rate * -2 * (y_i - (m.T.dot(x) + b)))
            fx_new = np.float64(m.T.dot(x) + b)
        if abs(fx_new - fx_old) < threshold:
            return m,b
        fx_old = fx_new
    return m,b

def linReg(X, y):
    m, b = gradient_descent(X, y, 0.000001, 0.1, 5000)
    
    return m, b

X_train, X_test, y_train, y_test, rawX_train, rawX_test, rawY_train, rawY_test = preprocess()
m,b = linReg(rawX_train, rawY_train)
m = m.reshape((-1, 1))

y_pred = np.dot(rawX_train,m) + b
varExplained = r2_score(rawY_train, y_pred)
print(varExplained)
