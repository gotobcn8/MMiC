
import torch
import numpy as np

def Rotate(dataset,mix_rate = 1.0,rotate_times = 1,dims = (2,3)):
    size = int(X.shape[0] * mix_rate)
    X,y = dataset['x'],dataset['y']
    X_selected_data = X[:size]
    y_selected_data = y[:size]
    augmentation_data = np.rot90(X_selected_data,k = rotate_times,dims=dims)
        
    return np.concatenate((X,augmentation_data),axis = 0),np.concatenate((y,y_selected_data),axis=0)