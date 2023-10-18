This repository trains univariate and multivariate linear regression models. 

The model is currently trained on 'Concrete_Data.csv', and to change the input file, you can do so in preprocess.py. 


FILES TO RUN: multiregression.py, uniregression.py


preprocess.py
    -Parses .csv file, and returns raw data that has been processed by featureNorm.py

featureNorm.py
    -Log transforms the features.

multiregression.py
    -Performs multivariate linear regression. 
    -The preprocess() function returns different subsets of data, including:
        -Raw Training Data Features
        -Raw Test Data Features 
        -Processed Training Data Features
        -Processed Test Data Features
        -Y Labels
    -On Line 53, 
        -m,b = linReg(rawX_train, rawY_train)
        -The parameters for linReg can be changed to train model on different training/test data. 
    -On Line 55, this uses the trained model on set of features.
        -y_pred = np.dot(rawX_train,m) + b
        -The variable in place of 'rawX_train' could be changed to test the model on different sets of features.
    -TO RUN THE CODE: 
        -python multiregression.py

uniregression.py
    -Performs univariate linear regression. 
    -All given scenarios for training and testing is coded in this file.
    -TO RUN THE CODE: 
        -python multiregression.py
