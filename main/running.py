



"""
Running algorithm to search best fitting model


Next steps:
    - Try meta features
    - Consistency check
    - Feature interaction / polynomial
    - Using era ?
    - hardcore EDA
"""



#=========================================================================================================
#================================ 0. MODULE


# Class
from numerai import Numerai

# Architecture
from architecture import models


#=========================================================================================================
#================================ 1. CLASS


stacking = Numerai(week=109)


#=========================================================================================================
#================================ 2. FEATURE ENGINEERING


# metafeature = ['variance', 'mean', 'distance']
# stages = [2, 0, 2]
# stacking.add_metafeature(metafeature, stages)


#=========================================================================================================
#================================ 3. TRAINING MODEL


###### MACHINE LEARNING
stacking.add_model(models)
stacking.fit_tune(nCores, neuralNetworkCompiler=False, evaluate=True, interaction=None)


###### DEEP LEARNING
# stacking.neuralNetworkCompiler(learningRate=0.0001, batch=64, epoch=2, cvNumber=2, displayStep=10000, evaluate=True, useGPU=True)


#=========================================================================================================
#================================ 4. PREDICTION


# stacking.submit(nCores=-1, submissionNumber=2, week=109)


