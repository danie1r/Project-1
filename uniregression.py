import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocess import preprocess
from sklearn.metrics import r2_score
import seaborn as sns

# This function performs gradient descent and updates the weight parameters for each features. 
# Input: 
#   X - features matrix
#   y - labels vector, or true output
#   learning_rate - The hyperparameter learning rate for the gradient descent. 
#   alpha - Hyperparameter for L2 regularization. Constant for the regularizer
# Output:
#   m - The resulting vector of weight parameters for each feature.
#   b - The resulting bias parameter.
def gradient_descent(x, y, learning_rate, alpha, max_iter):
    threshold = 0.0001
    m = np.random.randn(1)
    b = np.random.randn(1)
    fx_old = np.float64('inf')
    for iteration in range(max_iter):
        fx_new = 0
        for i in range(len(x)):
            index = np.random.randint(len(x))
            x_i = x[index]
            y_i = y[index]
            gradient = np.float64(-2 * x_i * (y_i - (m * x_i + b)) + 2*alpha*m)
            m = np.float64(m - learning_rate * gradient)
            b = np.float64(b - learning_rate * -2 * (y_i - (m * x_i + b)))
            fx_new = np.float64(m * x_i + b)
        if abs(fx_new - fx_old) < threshold:
            return m,b
        fx_old = fx_new
    return m,b

# This function performs multivariate linear regression.The hyperparameters include learning_rate, regularizer constant.
# It also specifies the maximum number of iterations of gradient descent to be repeated.
# Input:
#   X - features matrix
#   y - vector of y-labels, true label
# Output:
#   m - The resulting vector of weight parameters for each feature.
#   b - The resulting bias parameter.
def linReg(X, y):
    
    num_features = len(X[0])
    m = []
    b = []
    for i in range(num_features):
        temp_m, temp_b = gradient_descent(X[:,i], y, 0.000001, 0.1, 5000)
        m.append(temp_m)
        b.append(temp_b)
    
    return m, b


X_train, X_test, y_train, y_test, rawX_train, rawX_test, rawY_train, rawY_test = preprocess()


# Train univariate model on preprocessed training data.
m,b = linReg(X_train, y_train)

# # univariate model on training data (preprocessed) and tested on training data (preprocessed)
train_data_res_processed = []
for featureNum in range(8):
    y_pred = m[featureNum] * X_train[:,featureNum] + b[featureNum]
    varExplained = r2_score(y_train, y_pred)
    train_data_res_processed.append(varExplained)
    print(f"Var Explained: {varExplained}")
    plt.scatter(X_train[:,featureNum], y_train)
    plt.plot(X_train[:,featureNum], y_pred, 'r-')
    plt.show()
print("VarE of Univariate model on train(preprocessed): " + train_data_res_processed)

# # univariate model trained on training data (preprocessed) and tested on test data (preprocessed)
test_data_res_processed = []
for featureNum in range(8):
    y_pred = m[featureNum] * X_test[:,featureNum] + b[featureNum]
    varExplained = r2_score(y_test, y_pred)
    test_data_res_processed.append(varExplained)
print("VarE of Univariate model on test(preprocessed): " + test_data_res_processed)

# # Train univariate model on raw training data
m_raw, b_raw = linReg(rawX_train, rawY_train)

# # univariate model trained on training data (raw) and tested on training data (raw)
test_data_res_raw = []
for featureNum in range(8):
    y_pred = np.float64(m_raw[featureNum] * rawX_train[:,featureNum] + b_raw[featureNum])
    varExplained = r2_score(rawY_train, y_pred)
    test_data_res_raw.append(varExplained)
print("VarE of Univariate model on train(raw): " + test_data_res_raw)

# # univariate model trained on training data (raw) and tested on test data(raw).
for featureNum in range(8):
    y_pred = np.float64(m_raw[featureNum] * rawX_test[:,featureNum] + b_raw[featureNum])
    varExplained = r2_score(rawY_test, y_pred)
    test_data_res_raw.append(varExplained)

print("VarE of Univariate model on test(raw): " + test_data_res_raw)