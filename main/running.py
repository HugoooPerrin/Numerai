



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

# Architecture
from architecture import models, nCores


if __name__ == '__main__':

    #=========================================================================================================
    #================================ 1. CLASS


    stacking = Numerai(week=110)


    #=========================================================================================================
    #================================ 2. FEATURE ENGINEERING


    # metafeature = ['variance', 'mean', 'distance']
    # stages = [2, 0, 2]
    # stacking.add_metafeature(metafeature, stages)


    #=========================================================================================================
    #================================ 3. TRAINING MODEL


    stacking.add_model(models)
    stacking.training(nCores, stageNumber=1, neuralNetworkCompiler=False, evaluate=True)

    # Always compile with sklearn model before neural nets 
    stacking.compile(neuralNetworkCompiler=False)
    stacking.compile(neuralNetworkCompiler=True, learningRate=0.0001, batch=64, epoch=2, cvNumber=1, displayStep=10000, evaluate=True, useGPU=False):


    #=========================================================================================================
    #================================ 4. PREDICTION


    # stacking.submit(nCores=-1, submissionNumber=1, week=110)


