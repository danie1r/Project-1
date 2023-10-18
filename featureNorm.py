import numpy as np
import matplotlib.pyplot as plt


# This function pre-processes the raw feature data by log transofrming and adding 1. 1 is added to prevent 0 feature values. 
def featureNorm(rawX, rawY):
    processedX = np.array(rawX, dtype = np.float64)
    processedX = np.log(processedX + 1)
    processedY = np.array(rawY, dtype = np.float64)
    # processedY = np.log(processedY)
    return processedX, processedY