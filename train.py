import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn.functional as F
from dataset import get_array
from model import LSTM
from sklearn.model_selection import train_test_split
from tqdm import tqdm

x, y = get_array()
x_train, y_train, x_test, y_test = x[:-2], y[:-2], x[-1], y[-1]
print(x.shape, y.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train, x_test, y_train, y_test = torch.Tensor(x_train).to(device), torch.Tensor(x_test).to(device), torch.Tensor(y_train).to(device), torch.Tensor(y_test).to(device)

lr = 0.001
input_size = 1
hidden_size = 256
num_layers = 5
num_classes = 1
net = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device)
loss_function = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=lr)
epochs = 50

for epoch in range(epochs):
    for x, y in zip(x_train, y_train):
        x, y = x.to(device), y.to(device)
        outputs = net(x)
        # print(outputs.shape, y.shape)
        loss = loss_function(outputs.view(32), y.view(32))
        net.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        print(loss)

with torch.no_grad():
    output = net(x_test)
    for i, j in zip(output, y_test):
        print(i, j)
    print(loss_function(output, y_test))

# print(x_test, x_test.shape)
a = 16  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter
fig = plt.figure(figsize=(25, 75))
y_plot = []
for i, j, k in zip(x_test, y_test, output):
    i = i.view(5).tolist()
    j = j.tolist()
    k = k.tolist()
    plt.subplot(a, b, c)
    plt.plot(range(5), i)
    plt.scatter([5], j)
    plt.scatter([5], k, color="red")
    plt.ylim([0, 1])
    c += 1

# plt.tight_layout()
plt.show()
