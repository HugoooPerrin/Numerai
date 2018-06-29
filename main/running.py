




"""
Running algorithm to search best fitting model

"""




#=========================================================================================================
#================================ 0. MODULE


# Class
from numerai import Numerai

# Architecture
from architecture import models


##### RUNNING


if __name__ == '__main__':

    names = ['bernie', 'jordan', 'elizabeth', 'ken', 'charles']

    for name in ['elizabeth']: 

        print('\n----------------------------  {}  ----------------------------'.format(name.upper()))


    #=========================================================================================================
    #================================ 1. CLASS


        stacking = Numerai(week=113, name=name)

        stacking.load_data(stageNumber=1, evaluate=False, knn=False)


    #=========================================================================================================
    #================================ 2. FEATURE ENGINEERING


    ## Distance based features
        # stacking.kmeansTrick(k=5, stage=[1], interaction=False)
        # stacking.knnDistances(name=name, stage=[1], interaction=False) # X features precomputed

    ## Dimensionality reduction based features
        stacking.PCA(n_components=5, stage=[2], interaction=False)
        # stacking.autoEncoder(stage=[1], interaction=False,
        #                      layers=[25, 5, 25], dropout=0.6, learningRate=0.00002, batch=64, epoch=4,
        #                      cvNumber=3, displayStep=500, useGPU=True, evaluate=False)


    #=========================================================================================================
    #================================ 3. TRAINING MODEL

    ## Hardware
        nCores = 8
        useGPU = True


    ## DEEP LEARNING
        # stacking.trainingNN(layers=[55,20], dropout=0.7, learningRate=0.000002, batch=64, epoch=4,
        #                     cvNumber=3, displayStep=500, useGPU=useGPU, evaluate=True)


    ## MACHINE LEARNING
        stacking.training(nCores, models)


    #=========================================================================================================
    #================================ 4. COMPILATION


    ## MACHINE LEARNING
        stacking.compile(nCores, neuralNetworkCompiler=False)


    ## DEEP LEARNING
        # stacking.compile(nCores, neuralNetworkCompiler=True, 
        #                  hidden=20, dropout=0.5, learningRate=0.000001, batch=32, epoch=5, 
        #                  cvNumber=1, displayStep=2000, useGPU=useGPU)


    #=========================================================================================================
    #================================ 5. PREDICTION


        stacking.submit(submissionNumber=1, week=113)


