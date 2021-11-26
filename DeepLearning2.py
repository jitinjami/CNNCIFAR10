
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def train(model, train_loader, loss_fn, optimizer, device, num_epochs):
    # Training
    val_loss_list = np.zeros(num_epochs)
    train_loss_list = np.zeros(num_epochs)
    for epoch in range(1, num_epochs):
        running_loss = 0.0
        running_total = 0
        running_correct = 0
        run_step = 0
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            # shape of input images is (B, 1, 28, 28).
            images = images.to(device)
            labels = labels.to(device)  # shape (B).
            outputs = model(images)  # shape (B, 10).
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()  # reset gradients.
            loss.backward()  # compute gradients.
            optimizer.step()  # update parameters.

            running_loss += loss.item()
            running_total += labels.size(0)

            with torch.no_grad():
                _, predicted = outputs.max(1)
            running_correct += (predicted == labels).sum().item()
            run_step += 1
            if i % 200 == 0:
                # check accuracy.
                print(f'epoch: {epoch}, steps: {i}, '
                    f'train_loss: {running_loss / run_step :.3f}, '
                    f'running_acc: {100 * running_correct / running_total:.1f} %')
                running_loss = 0.0
                running_total = 0
                running_correct = 0
                run_step = 0
        
        train_loss_list[epoch] = 100 * running_correct / running_total
        # validate
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in valid_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_acc = 100 * correct / total
        print(f'Validation accuracy: {100 * correct / total} %')
        print(f'Validation error rate: {100 - 100 * correct / total: .2f} %')
        val_loss_list[epoch] = 100 - 100 * correct / total
    print('Finished Training')
    return [model, val_loss_list, train_loss_list]

def test(model, test_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_acc = 100 * correct / total
    print(f'Test accuracy: {100 * correct / total} %')
    print(f'Test error rate: {100 - 100 * correct / total: .2f} %')

    return (100 * correct / total)


class CNNet(nn.Module):
  def __init__(self):
    super(CNNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3) # output: 32 x 30 x 30
    self.conv2 = nn.Conv2d(32, 32, 3) #output: 32 x 28 x 28
    self.pool1 = nn.MaxPool2d(2, 2) #output: 32 x 14 x 14
    self.conv3 = nn.Conv2d(32, 64, 3) #output: 64 x 12 x 12
    self.conv4 = nn.Conv2d(64, 64, 3) #output: 64 x 10 x 10
    self.pool2 = nn.MaxPool2d(2, 2) #output: 64 x 5 x 5
    self.fc1 = nn.Linear(64 * 5 * 5, 512)
    self.fc2 = nn.Linear(512, 10)

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = self.pool1(out)
    out = F.relu(self.conv3(out))
    out = F.relu(self.conv4(out))
    out = self.pool2(out)
    out = out.view(-1, 64 * 5 * 5)
    out = F.relu(self.fc1(out))
    out = self.fc2(out)
    return out

class CNNet2(nn.Module):
  def __init__(self):
    super(CNNet2, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3) # output: 32 x 30 x 30
    self.conv2 = nn.Conv2d(32, 32, 3) #output: 32 x 28 x 28
    self.pool1 = nn.MaxPool2d(2, 2) #output: 32 x 14 x 14
    self.conv3 = nn.Conv2d(32, 64, 3) #output: 64 x 12 x 12
    self.conv4 = nn.Conv2d(64, 64, 3) #output: 64 x 10 x 10
    self.pool2 = nn.MaxPool2d(2, 2) #output: 64 x 5 x 5
    self.fc1 = nn.Linear(64 * 5 * 5, 512)
    self.fc2 = nn.Linear(512, 10)
    self.dropout1 = nn.Dropout2d(p=0.05)
    self.dropout2 = nn.Dropout(p=0.05)

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = self.pool1(out)
    out = self.dropout1(out)
    out = F.relu(self.conv3(out))
    out = F.relu(self.conv4(out))
    out = self.pool2(out)
    out = self.dropout1(out)
    out = out.view(-1, 64 * 5 * 5)
    out = F.relu(self.fc1(out))
    out = self.dropout2(out)
    out = self.fc2(out)
    return out
 
class CNNet3(nn.Module):
  def __init__(self, dp):
    super(CNNet3, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3) # output: 32 x 30 x 30
    self.conv2 = nn.Conv2d(32, 32, 3) #output: 32 x 28 x 28
    self.pool1 = nn.MaxPool2d(2, 2) #output: 32 x 14 x 14
    self.conv3 = nn.Conv2d(32, 64, 3) #output: 64 x 12 x 12
    self.conv4 = nn.Conv2d(64, 64, 3) #output: 64 x 10 x 10
    self.pool2 = nn.MaxPool2d(2, 2) #output: 64 x 5 x 5
    self.fc1 = nn.Linear(64 * 5 * 5, 512)
    self.fc2 = nn.Linear(512, 10)
    self.dropout1 = nn.Dropout2d(p=dp)
    self.dropout2 = nn.Dropout(p=dp)

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = self.pool1(out)
    out = self.dropout1(out)
    out = F.relu(self.conv3(out))
    out = F.relu(self.conv4(out))
    out = self.pool2(out)
    out = self.dropout1(out)
    out = out.view(-1, 64 * 5 * 5)
    out = F.relu(self.fc1(out))
    out = self.dropout2(out)
    out = self.fc2(out)
    return out


dir_path = os.path.dirname(os.path.realpath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

imgs = [item[0] for item in cifar_trainset] # item[0] and item[1] are image and its label
imgs = torch.stack(imgs, dim=0).numpy()

# calculate mean over each channel (r,g,b)
mean_r = imgs[:,0,:,:].mean()
mean_g = imgs[:,1,:,:].mean()
mean_b = imgs[:,2,:,:].mean()

# calculate std over each channel (r,g,b)
std_r = imgs[:,0,:,:].std()
std_g = imgs[:,1,:,:].std()
std_b = imgs[:,2,:,:].std()

batch_size = 32

my_transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean = [mean_r,mean_g,mean_b], std = [std_r,std_g,std_b]),
                              ])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=my_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=my_transform)

# Dataloaders
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=False)

idx = np.arange(len(train_set))
val_indices = idx[-1000:]
train_indices= idx[:-1000]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          sampler=train_sampler, num_workers=2)

valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          sampler=valid_sampler, num_workers=2)

#######################################
#### CNN1 - Varying learning rates ####
#######################################

learning_rates = [0.001,0.0015,0.002,0.0025,0.003,0.035,0.004]
trained_models1 = []
val_loss1 = []
train_loss1 = []
test_loss1 = []

for i in range(len(learning_rates)):
    momentum = 0.9

    model = CNNet()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates[i], momentum=momentum)

    num_epochs = 20

    [model_trained, val_loss, train_loss] = train(model, train_loader, loss_fn, optimizer, device, num_epochs)

    model_save_name = F'model1_{i}.pt'
    path = dir_path + F"/{model_save_name}" 
    torch.save(model_trained.state_dict(), path)

    d_model1 = {'Val_Loss': val_loss, 'Train_loss': train_loss}
    df_model1 = pd.DataFrame(data = d_model1)
    df_model1.to_csv(dir_path + F"/model1_{i}.csv")

    test_loss = test(model_trained, test_loader)
    test_loss1.append(test_loss)
    trained_models1.append(model_trained)
    val_loss1.append(val_loss)
    train_loss1.append(train_loss)

d_test1 = {'CNN1': test_loss1}
df_test1 = pd.DataFrame(data = d_test1)
df_test1.to_csv(dir_path + F"/test1.csv")

#######################################
#### CNN2 - Varying learning rates ####
#######################################

trained_models2 = []
val_loss2 = []
train_loss2 = []
test_loss2 = []

for i in range(len(learning_rates)):
    momentum = 0.9

    model = CNNet2()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates[i], momentum=momentum)

    num_epochs = 30

    [model_trained, val_loss, train_loss] = train(model, train_loader, loss_fn, optimizer, device, num_epochs)

    model_save_name = F'model2_{i}.pt'
    path = dir_path + F"/{model_save_name}" 
    torch.save(model_trained.state_dict(), path)

    d_model2 = {'Val_Loss': val_loss, 'Train_loss': train_loss}
    df_model2 = pd.DataFrame(data = d_model2)
    df_model2.to_csv(dir_path + F"/model2_{i}.csv")

    test_loss = test(model_trained, test_loader)
    test_loss2.append(test_loss)
    trained_models2.append(model_trained)
    val_loss2.append(val_loss)
    train_loss2.append(train_loss)

d_test2 = {'CNN2': test_loss2}
df_test2 = pd.DataFrame(data = d_test2)
df_test2.to_csv(dir_path + F"/test2.csv")

##############################################
#### CNN2 - Varying dropout probabilities ####
##############################################

learning_rate = 0.002
dp = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

trained_models3 = []
val_loss3 = []
train_loss3 = []
test_loss3 = []

for i in range(len(dp)):
    momentum = 0.9

    model = CNNet3(dp= dp[i])
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    num_epochs = 30

    [model_trained, val_loss, train_loss] = train(model, train_loader, loss_fn, optimizer, device, num_epochs)

    model_save_name = F'model3_{i}.pt'
    path = dir_path + F"/{model_save_name}" 
    torch.save(model_trained.state_dict(), path)

    d_model3 = {'Val_Loss': val_loss, 'Train_loss': train_loss}
    df_model3 = pd.DataFrame(data = d_model3)
    df_model3.to_csv(dir_path + F"/model3_{i}.csv")

    test_loss = test(model_trained, test_loader)
    test_loss3.append(test_loss)
    trained_models3.append(model_trained)
    val_loss3.append(val_loss)
    train_loss3.append(train_loss)

d_test3 = {'CNN3': test_loss3}
df_test3 = pd.DataFrame(data = d_test3)
df_test3.to_csv(dir_path + F"/test3.csv")

###################################
#### CNN4 - Best Case Scenario ####
###################################
learning_rate = 0.003
dp = 0.2

trained_models4 = []
val_loss4 = []
train_loss4 = []
test_loss4 = []

momentum = 0.9

model = CNNet3(dp= dp)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

num_epochs = 30

[model_trained, val_loss, train_loss] = train(model, train_loader, loss_fn, optimizer, device, num_epochs)

model_save_name = F'model_best.pt'
path = dir_path + F"/{model_save_name}" 
torch.save(model_trained.state_dict(), path)

d_model3 = {'Val_Loss': val_loss, 'Train_loss': train_loss}
df_model3 = pd.DataFrame(data = d_model3)
df_model3.to_csv(dir_path + F"/model_best.csv")

test_loss = test(model_trained, test_loader)
