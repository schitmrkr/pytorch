# MNIST
# DataLoader, Transformer
# Multilayer Neural Net, Activation function
# Loss and optimizer
# Training loop (Batch training)
# Model evaluation
# GPU support


# tensorboard --logdir=runs  #to start tensorboard

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist1")

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
input_size = 784  #28x28
hidden_size = 100
num_classes = 10
num_epoches = 2
batch_size = 100
learning_rate = 0.001

#MINST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(samples[i][0], cmap='gray')

#plt.show()

############ write to tensorboard  ##############
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist images', img_grid)
writer.close()
#sys.exit()

##################################################

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        #no softmax because we need to apply cross entropy using torch fucntion

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()  #this is the reason we didn't apply softmax to last layer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

############### tensorboard ################
writer.add_graph(model, samples.reshape(-1, 28*28))
writer.close()
#sys.exit()
############################################

#training loop
n_total_steps = len(train_loader)

running_loss = 0.0
running_correct = 0

for epoch in range(num_epoches):
    for i, (images, labels) in enumerate(train_loader):
        # change the shape of images from 100, 1, 28, 28
        # to 100, 784
        # labels are 1,2,3,4,5,6,7,8,9 not logits 

        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predictions = torch.max(outputs, 1)

        running_correct = (predictions == labels).sum().item()
        if (i+1) % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epoches}, step: {i+1}/{n_total_steps}, loss: {loss.item():.4f}')
            ################# tensor board ##############
            writer.add_scalar('training loss', running_loss/100, epoch * n_total_steps * i)
            writer.add_scalar('accuracy', running_correct/100, epoch * n_total_steps * i)
            running_loss = 0.0
            running_correct = 0
            ############################################

#test

labels_ = []
preds_ = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        class_predictions = [F.softmax(outputs, dim=0)]

        preds_.append(class_predictions)
        labels_.append(predictions)
    
    preds_ = torch.cat([torch.stack(batch) for batch in preds_])
    labels_ = torch.cat(labels_)

    acc = 100 * n_correct / n_samples
    print(f'accuracy = {acc}')

    ################## tensor board ####################
    classes = range(10)
    for i in classes:
        labels_i = (labels_ == i)
        preds_i = preds_[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
    
    ###################################################

    

