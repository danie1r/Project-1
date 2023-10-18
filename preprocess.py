from featureNorm import featureNorm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Function to preprocess the data
def preprocess():
    rawX = []
    rawY = []
    with open('Concrete_Data.csv', 'r', encoding='utf-8-sig') as f:
        ingest = f.read().splitlines()
        for v in ingest:
            temp_data = v.split(",")
            rawX.append(temp_data[0:len(temp_data)-1])
            rawY.append(temp_data[len(temp_data)-1:])
    rawX = np.array(rawX, dtype=np.float64)
    rawY = np.array(rawY, dtype=np.float64)
    
    processedX, processedY = featureNorm(rawX, rawY)
    
    # The train_test_split function from sklearn library splits the data into train and test data. Splits into 130 samples of test,
    # and others as train. The samples are divided randomly.
    X_train, X_test, y_train, y_test = train_test_split(processedX, processedY, test_size=130, random_state=42)
    rawX_train, rawX_test, rawY_train, rawY_test = train_test_split(rawX, rawY, test_size=130, random_state=42)
    return X_train, X_test, y_train, y_test, rawX_train, rawX_test, rawY_train, rawY_test