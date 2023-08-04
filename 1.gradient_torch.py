import torch 

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad = True)

def forward(x):
    return w * x

def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


predic = forward(5)
print(f'Prediction before training: f(5) = {predic:.3f}')

learning_rate = 0.01
n_iters = 20



for epoch in range(n_iters):

    y_pred = forward(X)

    l = loss(Y, y_pred)

    #backward pass and gradient computation
    l.backward()  #dl/dw

    #update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    #zero the gradiants
    w.grad.zero_()

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')


predic = forward(5)
print(f'Prediction after training: f(5) = {predic:.3f}')



