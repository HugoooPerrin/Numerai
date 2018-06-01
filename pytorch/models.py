

import torch
import torch.nn as nn

# --------------------------------------------------------------
# --------------------------------------------------------------


class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(15, 1))
        
    def forward(self, x):
        out = self.linear(x)
        return out



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
                        # nn.Conv1d(1, 10, kernel_size=1),       # (50-(1-1)) = 50
                        nn.Conv1d(1, 10, kernel_size=5),      # (50-(5-1)) = 46
                        # nn.Conv1d(20, 40, kernel_size=5),      # (46-(5-1)) = 41
                        # nn.BatchNorm1d(10),
                        nn.Tanh(),
                        nn.Dropout(0.5),
                        nn.MaxPool1d(2, 2))                   # |(46-2)/2+1| = 23

        self.fc = nn.Sequential(
                        nn.Linear(23*10, 75),
                        nn.Tanh(),
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
                            nn.Conv1d(1, 8, kernel_size=1),   # (50-(1-1)) = 50
                            nn.Conv1d(8, 16, kernel_size=5),  # (50-(5-1)) = 46
                            # nn.BatchNorm1d(32),
                            nn.Tanh(),
                            nn.Dropout(0.5),
                            nn.MaxPool1d(2, 2))                # |(46-2)/2+1| = 23 !

        self.module2 = nn.Sequential(
                            nn.Conv1d(1, 8, kernel_size=1),    # (50-(1-1)) = 50
                            nn.Conv1d(8, 16, kernel_size=10),  # (50-(10-1)) = 41
                            # nn.BatchNorm1d(32),
                            nn.Tanh(),
                            nn.Dropout(0.5),
                            nn.MaxPool1d(2, 2))                # |(41-2)/2+1| = 20 !

        self.module3 = nn.Sequential(
                            nn.Conv1d(1, 8, kernel_size=1),    # (50-(1-1)) = 50
                            nn.Conv1d(8, 16, kernel_size=20),  # (50-(20-1)) = 31
                            # nn.BatchNorm1d(32),
                            nn.Tanh(),
                            nn.Dropout(0.5),
                            nn.MaxPool1d(2, 2))                # |(31-2)/2+1| = 15 !

        self.final_module = nn.Sequential(
                                nn.Linear(23*16+20*16+15*16, 512),
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(512, 128),
                                nn.Tanh(),
                                nn.Dropout(0.5),
                                nn.Linear(128, 1))
        
    def forward(self, x):

        inter1 = self.module1(x)
        inter1 = inter1.view(-1, 23*16)

        inter2 = self.module2(x)
        inter2 = inter2.view(-1, 20*16)

        inter3 = self.module3(x)
        inter3 = inter3.view(-1, 15*16)

        out = torch.cat((inter1, inter2, inter3), 1)
        del inter1, inter2, inter3

        out = self.final_module(out)

        return out
