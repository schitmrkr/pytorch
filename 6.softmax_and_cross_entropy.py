import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x), axis=0)

x = np.array([2.0,1.0,0.1])
outputs = softmax(x)
print(f'softmax output: {outputs}')

x = torch.tensor([2.0,1.0,0.1])
outputs = torch.softmax(x, dim=0)
print(f'softmax output: {outputs}')



def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

#y must be one hot encoded
# if class 0: [1,0,0]
# if class 1: [0,1,0]
# if class 2: [0,0,1]

Y = np.array([1, 0, 0])

#y_pred has probabilities
Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')


#in pytorch
