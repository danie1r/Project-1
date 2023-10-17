import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocess import preprocess
from sklearn.metrics import r2_score

# f(x) = mx + b
# m_new = m_old - alpha/n * -2x(y-(m_old*x + b_old))
# b_new = b_old - alpha/n * -2(y - (m_old*x + b_old))
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

m,b = linReg(X_train, y_train)

# univariate model on training data (preprocessed)
# train_data_res_processed = []
# for featureNum in range(8):
#     y_pred = m[featureNum] * X_train[:,featureNum] + b[featureNum]
#     varExplained = r2_score(y_train, y_pred)
#     train_data_res_processed.append(varExplained)
#     print(f"Var Explained: {varExplained}")
#     plt.scatter(X_train[:,featureNum], y_train)
#     plt.plot(X_train[:,featureNum], y_pred, 'r-')
#     plt.show()
# print(train_data_res_processed)
# test_data_res_processed = []
# for featureNum in range(8):
#     y_pred = m[featureNum] * X_test[:,featureNum] + b[featureNum]
#     varExplained = r2_score(y_test, y_pred)
#     test_data_res_processed.append(varExplained)
# print(test_data_res_processed)

# m_raw, b_raw = linReg(rawX_train, rawY_train)
# test_data_res_raw = []
# for featureNum in range(8):
#     y_pred = np.float64(m_raw[featureNum] * rawX_train[:,featureNum] + b_raw[featureNum])
#     varExplained = r2_score(rawY_train, y_pred)
#     test_data_res_raw.append(varExplained)

# print(test_data_res_raw)
# for featureNum in range(8):
#     y_pred = np.float64(m_raw[featureNum] * rawX_test[:,featureNum] + b_raw[featureNum])
#     varExplained = r2_score(rawY_test, y_pred)
#     test_data_res_raw.append(varExplained)

# print(test_data_res_raw)