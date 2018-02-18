

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms



#--------------------------------------------------------------
#--------------------------------------------------------------

class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(50, 200),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(100, 1))
        
    def forward(self, x):
        out = self.linear(x)
        return out


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
                        nn.Conv1d(1, 5, kernel_size=1),  # (50-(1-1))*5 = 50*5
                        nn.Conv1d(5, 10, kernel_size=5),  # (50-(5-1))*10 = 46*10
                        # nn.Conv1d(60, 120, kernel_size=5),  # (46-(5-1))*120 = 41*120
                        nn.BatchNorm1d(10),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.MaxPool1d(2, 2))              # |(46-2)/2+1|*10 = 23*10

        self.fc = nn.Sequential(
                        nn.Linear(23*10, 75),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(75, 1))
        
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(-1, 23*10)
        out = self.fc(out)
        return out


class Inception(nn.Module):

    def __init__(self):
        super(Inception, self).__init__()

        self.module1 = nn.Sequential(
                            nn.Conv1d(1, 16, kernel_size=1),  # (50-(1-1))*16 = 50*16
                            nn.Conv1d(16, 32, kernel_size=5),  # (50-(5-1))*32 = 46*32
                            nn.BatchNorm1d(32),
                            nn.Tanh(),
                            nn.Dropout(0.5),
                            nn.MaxPool1d(2, 2))              # |(46-2)/2+1|*32 = 23*32

        self.module2 = nn.Sequential(
                            nn.Conv1d(1, 8, kernel_size=1),  # (50-(1-1))*8 = 50*8
                            nn.Conv1d(8, 16, kernel_size=10),  # (50-(10-1))*16 = 41*16
                            nn.BatchNorm1d(16),
                            nn.Tanh(),
                            nn.Dropout(0.5),
                            nn.MaxPool1d(2, 2))              # |(41-2)/2+1|*16 = 20*16

        self.module3 = nn.Sequential(
                            nn.Conv1d(1, 8, kernel_size=1),  # (50-(1-1))*8 = 50*8
                            nn.Conv1d(8, 16, kernel_size=20),# (50-(20-1))*16 = 31*16
                            nn.BatchNorm1d(16),
                            nn.Tanh(),
                            nn.Dropout(0.5),
                            nn.MaxPool1d(2, 2))              # |(31-2)/2+1|*16 = 15*16

        self.final_module = nn.Sequential(
                                nn.Linear(23*32+20*16+15*16, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512, 1))
        
    def forward(self, x):

        inter1 = self.module1(x)
        inter1 = inter1.view(-1, 23*32)

        inter2 = self.module2(x)
        inter2 = inter2.view(-1, 20*16)

        inter3 = self.module3(x)
        inter3 = inter3.view(-1, 15*16)

        out = torch.cat((inter1, inter2, inter3), 1)
        del inter1, inter2, inter3

        out = self.final_module(out)

        return out