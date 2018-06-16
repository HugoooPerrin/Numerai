




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

    names = ['bernie', 'jordan', 'ken', 'elizabeth', 'charles']

    for name in ['elizabeth']: #names:

        print('\n----------------------------  {}  ----------------------------'.format(name.upper()))


    #=========================================================================================================
    #================================ 1. CLASS


        stacking = Numerai(week=111, name=name)

        stacking.load_data(stageNumber=1, evaluate=False)


    #=========================================================================================================
    #================================ 2. FEATURE ENGINEERING


        stacking.kmeansTrick(k=15, stage=[2], interaction=False)
        stacking.PCA(n_components=2, stage=[1], interaction=True)
        # stacking.meanEncoding()
        # stacking.autoEncoder(stage=[1], interaction=False,
        #                      layers=[25, 10, 25], dropout=0.5, learningRate=0.00002, batch=64, epoch=4,
        #                      cvNumber=3, displayStep=500, useGPU=True, evaluate=False)


    #=========================================================================================================
    #================================ 3. TRAINING MODEL

    ## Hardware
        nCores = 6
        useGPU = True


    ## DEEP LEARNING
        # stacking.trainingNN(layers=[160,50], dropout=0.6, learningRate=0.00001, batch=64, epoch=5,
        #                     cvNumber=3, displayStep=500, useGPU=useGPU, evaluate=False)


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


        stacking.submit(submissionNumber=2, week=111)


