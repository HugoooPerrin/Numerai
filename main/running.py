



"""
Running algorithm to search best fitting model
"""



#=========================================================================================================
#================================ 0. MODULE


# Class
from numerai import Numerai

# Architecture
from architecture import models


try:
    import torch
    import torch.nn as nn
except:
    pass


if __name__ == '__main__':


    #=========================================================================================================
    #================================ 1. CLASS


    stacking = Numerai(week=111, name='jordan')

    # Loading data
    stacking.load_data(stageNumber=1, evaluate=True)


    #=========================================================================================================
    #================================ 2. FEATURE ENGINEERING


    stacking.kmeansTrick(k=3, stage=[1], interaction=True)
    # stacking.kernelPCA(n_components=3, kernel='rbf', stage=[1], interaction=True)


    #=========================================================================================================
    #================================ 3. TRAINING MODEL


    ## DEEP LEARNING
    # class NN(nn.Module):

    #     def __init__(self):
    #         super(NN, self).__init__()

    #         self.linear = nn.Sequential(
    #             nn.Linear(55, 25),
    #             nn.ReLU(),
    #             nn.Dropout(0.5),
    #             nn.Linear(25, 1))

    #     def forward(self, x):
    #         out = self.linear(x)
    #         return out
        
    # stacking.trainingNN(architecture=NN(), learningRate=0.00003, batch=64, epoch=6,
    #                        cvNumber=3, displayStep=1000, useGPU=True, evaluate=False)


    ## MACHINE LEARNING
    nCores = 6

    stacking.training(nCores, models)


    #=========================================================================================================
    #================================ 4. COMPILATION

    ## MACHINE LEARNING
    stacking.compile(nCores, neuralNetworkCompiler=False)


    ## DEEP LEARNING
    # class NN(nn.Module):

    # def __init__(self):
    #     super(NN, self).__init__()

    #     self.linear = nn.Sequential(
    #         nn.Linear(3, 3),
    #         nn.ReLU(),
    #         nn.Dropout(0.3),
    #         nn.Linear(3, 1))

    # def forward(self, x):
    #     out = self.linear(x)
    #     return out

    # stacking.compile(nCores, neuralNetworkCompiler=True, architectureNN=NN(), 
    #                  learningRate=0.000001, batch=32, epoch=5, cvNumber=1, 
    #                  displayStep=2000, useGPU=True)

    #=========================================================================================================
    #================================ 5. PREDICTION


    stacking.submit(submissionNumber=1, week=111)


