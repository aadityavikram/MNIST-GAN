import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ip=100, op=28 * 28):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(ip, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, op)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)  # negative slope = 0.2
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc4(x)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, ip=28 * 28, op=1):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(ip, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, op)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, 0.3)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, 0.3)
        x = self.fc3(x)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, 0.3)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x
