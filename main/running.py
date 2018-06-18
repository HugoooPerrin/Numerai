




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

    for name in ['elizabeth']: #names:

        print('\n----------------------------  {}  ----------------------------'.format(name.upper()))


    #=========================================================================================================
    #================================ 1. CLASS


        stacking = Numerai(week=112, name=name)

        stacking.load_data(stageNumber=1, evaluate=True)


    #=========================================================================================================
    #================================ 2. FEATURE ENGINEERING


    ## Distance based features
        stacking.kmeansTrick(k=10, stage=[1], interaction=False)
        stacking.knnDistances(k=[10, 50], stage=[1], interaction=False)

    ## Dimensionality reduction based features
        stacking.PCA(n_components=5, stage=[1], interaction=False)
        stacking.autoEncoder(stage=[1], interaction=False,
                             layers=[25, 10, 25], dropout=0.6, learningRate=0.00002, batch=64, epoch=4,
                             cvNumber=3, displayStep=500, useGPU=True, evaluate=False)


    #=========================================================================================================
    #================================ 3. TRAINING MODEL

    ## Hardware
        nCores = 6
        useGPU = True


    ## DEEP LEARNING
        stacking.trainingNN(layers=[75,20], dropout=0.6, learningRate=0.000005, batch=64, epoch=4,
                            cvNumber=3, displayStep=500, useGPU=useGPU, evaluate=False)


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


        stacking.submit(submissionNumber=1, week=112)


