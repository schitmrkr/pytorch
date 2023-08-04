#  1) Design our model (input, output size, forward_pass)
#  2) Construct loss and the optimizer
#  3) Training loop
#    - forward pass: compute prediction
#    - backward pass: compute gradiant
#    - update weights

import torch 
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([[5]], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features   #same as for x

w = torch.tensor(0.0, dtype=torch.float32, requires_grad = True)

#model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

predic = model(X_test)
print(f'Prediction before training: f(5) = {predic.item():.3f}')

learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):

    y_pred = model(X)

    #Loss
    l = loss(Y, y_pred)

    #backward pass and gradient computation
    l.backward()  #dl/dw

    #update weights
    optimizer.step()

    #zero the gradiants
    optimizer.zero_grad()

    if epoch % 2 == 0:
        [w, b] = model.parameters()
        print(w, b)
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')


predic = model(X_test)
print(f'Prediction after training: f(5) = {predic.item():.3f}')



