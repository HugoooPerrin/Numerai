



"""
Running algorithm to search best fitting model


Next steps:
    - NN architecture more flexible
    - Try meta features
    - Feature interaction / polynomial
    - Using era ?
    - hardcore EDA
"""



#=========================================================================================================
#================================ 0. MODULE


# Class
from numerai import Numerai


try:
    import torch
    import torch.nn as nn
except:
    pass


if __name__ == '__main__':


    #=========================================================================================================
    #================================ 1. CLASS


    stacking = Numerai(week=111)
    nCores = -1


    #=========================================================================================================
    #================================ 2. FEATURE ENGINEERING


    # metafeature = ['variance', 'mean', 'distance']
    # stages = [2, 0, 2]
    # stacking.add_metafeature(metafeature, stages)


    #=========================================================================================================
    #================================ 3. TRAINING MODEL

    stacking.add_model(models)

    # class NN(nn.Module):

    #     def __init__(self):
    #         super(NN, self).__init__()

    #         self.linear = nn.Sequential(
    #             nn.Linear(38, 10),
    #             nn.ReLU(),
    #             nn.Dropout(0.5),
    #             nn.Linear(10, 1))
            
    #     def forward(self, x):
    #         out = self.linear(x)
    #         return out

    # stacking.trainingNN(architecture=NN(), learningRate=0.0001, batch=64, epoch=5, 
    #                     cvNumber=1, displayStep=1000, evaluate=True, useGPU=True)

    stacking.training(nCores, stageNumber=1, evaluate=False)

    stacking.compile(nCores, neuralNetworkCompiler=False, evaluate=False)

    # stacking.compile(nCores, neuralNetworkCompiler=True, architecture=NN(), 
    #                  learningRate=0.0001, batch=64, epoch=2, cvNumber=1, 
    #                  displayStep=1000, evaluate=False, useGPU=True)


    #=========================================================================================================
    #================================ 4. PREDICTION


    stacking.submit(submissionNumber=1, week=111)


