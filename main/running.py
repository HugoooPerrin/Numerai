



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

    names = ['bernie', 'jordan', 'ken', 'elizabeth', 'charles']

    for name in ['ken', 'elizabeth', 'charles']: #names:

        print('\n---------------------------- {} ----------------------------'.format(name))

    #=========================================================================================================
    #================================ 1. CLASS


        stacking = Numerai(week=111, name=name)

        stacking.load_data(stageNumber=1, evaluate=False)


    #=========================================================================================================
    #================================ 2. FEATURE ENGINEERING


        stacking.kmeansTrick(k=15, stage=[2], interaction=False)
        stacking.PCA(n_components=2, stage=[1], interaction=True)
        # stacking.meanEncoding()
        # stacking.autoEncoder()

    #=========================================================================================================
    #================================ 3. TRAINING MODEL


    ## DEEP LEARNING
    # class NN(nn.Module):

    #     def __init__(self):
    #         super(NN, self).__init__()

    #         self.linear = nn.Sequential(
    #             nn.Linear(200, 20),
    #             nn.ReLU(),
    #             nn.Dropout(0.4),
    #             nn.Linear(20, 1))

    #     def forward(self, x):
    #         out = self.linear(x)
    #         return out
        
        # stacking.trainingNN(architecture=NN(), learningRate=0.000005, batch=64, epoch=5,
        #                        cvNumber=1, displayStep=1000, useGPU=True, evaluate=True)


    ## MACHINE LEARNING
        nCores = 3

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


