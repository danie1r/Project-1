import numpy as np
import matplotlib.pyplot as plt


def featureNorm(rawX, rawY):
    processedX = np.array(rawX, dtype = np.float64)
    processedX = np.log(processedX + 1)
    processedY = np.array(rawY, dtype = np.float64)
    # processedY = np.log(processedY)
    
    return processedX, processedY