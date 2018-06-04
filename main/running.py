



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
    # stacking.compile()


    #=========================================================================================================
    #================================ 4. PREDICTION


    # stacking.submit(nCores=-1, submissionNumber=1, week=110)


