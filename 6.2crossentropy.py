import torch
import torch.nn as nn
import numpy as np


#should not be one hot encoded for Y

loss = nn.CrossEntropyLoss()


# 3 samples
Y = torch.tensor([2,0,1])

# n_samples x n_classes = 1 x 3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1], [0.1, 1.0, 2.0]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3], [0.3, 0.5, 2.0], [2.0, 0.5, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, prediction_1 = torch.max(Y_pred_good, 1)
_, prediction_2 = torch.max(Y_pred_bad, 1)

print(prediction_1)
print(prediction_2)