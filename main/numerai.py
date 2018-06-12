





"""
Main class designed to quickly evaluate different model architectures over Numerai dataset

"""




#=========================================================================================================
#============================================ MODULE =====================================================


# General
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from copy import deepcopy

# Preprocessing
from scipy.spatial.distance import euclidean
from sklearn.decomposition import KernelPCA
from sklearn import cluster

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Deep learning
try:
    sys.path.append('../pytorch/')
    from utils import trainNN, predictNN
    import torch.utils.data as utils
    import torch.optim as optim
    import torch.nn as nn
    import torch
except:
    pass

# Utils
def diff(t_a, t_b):
    t_diff = relativedelta(t_a, t_b)
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

# Architecture
from architecture import models

#=========================================================================================================
#============================================ NUMERAI CLASS ==============================================


class Numerai(object):



    def __init__(self, week, name):

        self.week = week
        self.type = name
        self.kmeanStage = False

        self.modelNames = []
        self.models = []
        self.parameters = []
        self.baggingSteps = []
        self.nFeatures = []
        self.stage = []

        self.notYetNN_comp = True
        self.notYetNN_train = True


#----------------------------------------------------------------------------------------------
#-------------------------------------------- DATA --------------------------------------------


    def load_data(self, stageNumber, evaluate):
        print('\n---------------------------------------------')
        print('>> Loading data', end='...')
        Xtrain = pd.read_csv("../../../Datasets/Numerai/w{}/numerai_training_data.csv".format(self.week))
        Xvalid = pd.read_csv("../../../Datasets/Numerai/w{}/numerai_tournament_data.csv".format(self.week))
 
        # Xtrain = pd.read_csv("../../Data/numerai_training_data.csv")
        # Xvalid = pd.read_csv("../../Data/numerai_tournament_data.csv")

        self.evaluate = evaluate
        self.stageNumber = stageNumber

        real_data = Xvalid.copy(True)
        self.ids = Xvalid['id']

        Xvalid = Xvalid[Xvalid['data_type'] == 'validation']

        Ytrain = Xtrain['target_{}'.format(self.type)]
        Yvalid = Xvalid['target_{}'.format(self.type)]

        Xtrain.drop(['id', 'era', 'data_type', 'target_bernie', 'target_charles', 'target_elizabeth', 'target_jordan', 'target_ken'], inplace = True, axis = 1)
        Xvalid.drop(['id', 'era', 'data_type', 'target_bernie', 'target_charles', 'target_elizabeth', 'target_jordan', 'target_ken'], inplace = True, axis = 1)
        real_data.drop(['id', 'era', 'data_type', 'target_bernie', 'target_charles', 'target_elizabeth', 'target_jordan', 'target_ken'], inplace = True, axis = 1)

        if stageNumber == 1:
            if self.evaluate:
                # Xtrain1 = 40%, Xtrain2 = 40%, Xtest = 20%
                Xtrain1, Xtrain2, Ytrain1, Ytrain2 = train_test_split(Xtrain, Ytrain, test_size=0.6)
                Xtrain2, Xtest, Ytrain2, Ytest = train_test_split(Xtrain2, Ytrain2, test_size=0.33)
                print('done')
                print('\nXtrain1: {}'.format(Xtrain1.shape),
                      '\nYtrain1: {}'.format(Ytrain1.shape),
                      '\nXtrain2: {}'.format(Xtrain2.shape),
                      '\nYtrain2: {}'.format(Ytrain2.shape),
                      '\nXtest: {}'.format(Xtest.shape),
                      '\nYtest: {}'.format(Ytest.shape),
                      '\nXvalid: {}'.format(Xvalid.shape),
                      '\nYvalid: {}'.format(Yvalid.shape),
                      '\nSubmit data: {}\n'.format(real_data.shape))

                self.Xtrain = {1: Xtrain1,
                               2: Xtrain2,
                               'test': Xtest,
                               'valid': Xvalid,
                               'real_data': real_data}

                self.Ytrain = {1: Ytrain1,
                               2: Ytrain2,
                               'test': Ytest,
                               'valid': Yvalid}

            else:
                # Xtrain1 = 50%, Xtrain2 = 50%
                Xtrain1, Xtrain2, Ytrain1, Ytrain2 = train_test_split(Xtrain, Ytrain, test_size=0.5)
                print('done')
                print('\nXtrain1: {}'.format(Xtrain1.shape),
                      '\nYtrain1: {}'.format(Ytrain1.shape),
                      '\nXtrain2: {}'.format(Xtrain2.shape),
                      '\nYtrain2: {}'.format(Ytrain2.shape),
                      '\nXvalid: {}'.format(Xvalid.shape),
                      '\nYvalid: {}'.format(Yvalid.shape),
                      '\nSubmit data: {}\n'.format(real_data.shape))

                self.Xtrain = {1: Xtrain1,
                               2: Xtrain2,
                               'real_data': real_data}

                self.Ytrain = {1: Ytrain1,
                               2: Ytrain2}

        elif stageNumber == 2:
            if self.evaluate:
                # Xtrain1 = 30%, Xtrain2 = 30%, Xtrain3 = 30%, Xtest = 10%
                Xtrain1, Xtrain2, Ytrain1, Ytrain2 = train_test_split(Xtrain, Ytrain, test_size=0.7)
                Xtrain2, Xtrain3, Ytrain2, Ytrain3 = train_test_split(Xtrain2, Ytrain2, test_size=0.57)
                Xtrain3, Xtest, Ytrain3, Ytest = train_test_split(Xtrain3, Ytrain3, test_size=0.25)
                print('done')
                print('\nXtrain1: {}'.format(Xtrain1.shape),
                      '\nYtrain1: {}'.format(Ytrain1.shape),
                      '\nXtrain2: {}'.format(Xtrain2.shape),
                      '\nYtrain2: {}'.format(Ytrain2.shape),
                      '\nXtrain3: {}'.format(Xtrain3.shape),
                      '\nYtrain3: {}'.format(Ytrain3.shape),
                      '\nXtest: {}'.format(Xtest.shape),
                      '\nYtest: {}'.format(Ytest.shape),
                      '\nXvalid: {}'.format(Xvalid.shape),
                      '\nYvalid: {}'.format(Yvalid.shape),
                      '\nSubmit data: {}\n'.format(real_data.shape))

                self.Xtrain = {1: Xtrain1,
                               2: Xtrain2,
                               3: Xtrain3,
                               'test': Xtest,
                               'valid': Xvalid,
                               'real_data': real_data}

                self.Ytrain = {1: Ytrain1,
                               2: Ytrain2,
                               3: Ytrain3,
                               'test': Ytest,
                               'valid': Yvalid}
            else:
                # Xtrain1 = 33%, Xtrain2 = 33%, Xtrain3 = 33%
                Xtrain1, Xtrain2, Ytrain1, Ytrain2 = train_test_split(Xtrain, Ytrain, test_size=0.67)
                Xtrain2, Xtrain3, Ytrain2, Ytrain3 = train_test_split(Xtrain2, Ytrain2, test_size=0.5)
                print('done')
                print('\nXtrain1: {}'.format(Xtrain1.shape),
                      '\nYtrain1: {}'.format(Ytrain1.shape),
                      '\nXtrain2: {}'.format(Xtrain2.shape),
                      '\nYtrain2: {}'.format(Ytrain2.shape),
                      '\nXtrain3: {}'.format(Xtrain3.shape),
                      '\nYtrain3: {}'.format(Ytrain3.shape),
                      '\nXvalid: {}'.format(Xvalid.shape),
                      '\nYvalid: {}'.format(Yvalid.shape),
                      '\nSubmit data: {}\n'.format(real_data.shape))

                self.Xtrain = {1: Xtrain1,
                               2: Xtrain2,
                               3: Xtrain3,
                               'real_data': real_data}

                self.Ytrain = {1: Ytrain1,
                               2: Ytrain2,
                               3: Ytrain3}


#----------------------------------------------------------------------------------------------
#------------------------------------- FEATURE ENGINEERING ------------------------------------


    def kmeansTrick(self, k, stage=[1], interaction=False):


        print('\n---------------------------------------------')
        print('>> Processing Kmeans ------\n')

        self.kmeanStage = stage
        self.kmeansInteraction = interaction

    # Unsupervised learning
        print('Fitting model', end='...')
        model = cluster.KMeans(n_clusters=k, precompute_distances=False)
        model.fit(self.Xtrain[1])
        print('done')

    # Feature engineering
        print('Generating kmeans features', end='...')
        self.kmeanDist = {}
        for dataset in self.Xtrain:
            self.kmeanDist[dataset] = model.transform(self.Xtrain[dataset])
            self.kmeanDist[dataset] = pd.DataFrame(self.kmeanDist[dataset], columns=['kmeans{}'.format(i) for i in range(k)])
        print('done')



    def PCA(self, n_components, kernel='rbf', stage=[1], interaction=False):

        print('\n---------------------------------------------')
        print('>> Processing KernelPCA ------\n')

        self.pcaStage = stage
        self.pcaInteraction = interaction

    # Unsupervised learning
        print('Fitting model', end='...')
        model = KernelPCA(n_components=n_components, kernel='rbf')
        model.fit(self.Xtrain[1])
        print('done')

    # Feature engineering
        print('Generating KernelPCA features', end='...')
        self.KernelPCA = {}
        for dataset in self.Xtrain:
            self.KernelPCA[dataset] = model.transform(self.Xtrain[dataset])
            self.KernelPCA[dataset] = pd.DataFrame(self.KernelPCA[dataset], columns=['kernelPCA{}'.format(i) for i in range(n_components)])
        print('done')



    def meanEncoding(self, n, stage, interaction=False):
        pass



    def autoEncoder(self, n, stage, interaction=False):
        pass


#----------------------------------------------------------------------------------------------
#------------------------------------------- MODELS -------------------------------------------


    def trainingNN(self, architecture, learningRate=0.0001, batch=64, epoch=5, 
                   cvNumber=1, displayStep=1000, useGPU=True, evaluate=True):

        self.notYetNN_train = False
        
    # Data
        self.firstStagePrediction = {}
        for dataset in self.Xtrain:
            self.firstStagePrediction[dataset] = pd.DataFrame()
            
        print('\n---------------------------------------------')
        print('>> Processing Neural Network ------\n')

        XtrainNNData = {}
        YtrainNNData = {}
        for dataset in self.Xtrain:

        # Kmeans trick ------------ 

            if 1 in self.kmeanStage:
                if self.kmeansInteraction:
                    XtrainNNData[dataset] = pd.DataFrame()   # Dataframe creation
                    for feature in self.Xtrain[dataset].columns:
                        for meta in self.kmeanDist[dataset].columns:
                            XtrainNNData[dataset]['{}_{}'.format(feature, meta)] = self.Xtrain[dataset][feature].values * self.kmeanDist[dataset][meta].values
                else:
                    XtrainNNData[dataset] = pd.concat([self.Xtrain[dataset].reset_index(drop=True),
                                                       self.kmeanDist[dataset].reset_index(drop=True)], axis=1)

        # Kernel PCA --------------

            if 1 in self.pcaStage:
                if self.pcaInteraction:
                    for feature in self.Xtrain[dataset].columns:
                        for meta in self.KernelPCA[dataset].columns:
                            XtrainNNData[dataset]['{}_{}'.format(feature, meta)] = self.Xtrain[dataset][feature].values * self.KernelPCA[dataset][meta].values
                else:
                    XtrainNNData[dataset] = pd.concat([self.Xtrain[dataset].reset_index(drop=True),
                                                       self.KernelPCA[dataset].reset_index(drop=True)], axis=1)

        # No metafeature ----------

            else:
                XtrainNNData[dataset] = self.Xtrain[dataset]

        # To array
        for dataset in self.Xtrain:
            XtrainNNData[dataset] = np.array(XtrainNNData[dataset])
            if dataset != 'real_data':
                YtrainNNData[dataset] = np.array(self.Ytrain[dataset].values)
                YtrainNNData[dataset] = YtrainNNData[dataset].reshape(YtrainNNData[dataset].shape[0], 1)

        XtrainNNData['nn'] = np.concatenate((XtrainNNData[1], XtrainNNData['test']), axis=0)
        YtrainNNData['nn'] = np.concatenate((YtrainNNData[1], YtrainNNData['test']), axis=0)

        if evaluate:
        # Tuning hyperparameter
            cvScore = 0

            time = datetime.now()

            for i in range(cvNumber):
                print('\nLoop number {}'.format(i+1))

                Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(XtrainNNData['nn'], YtrainNNData['nn'], test_size=0.25)

            # Tensor

                train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(Xtrain), 
                                                               torch.FloatTensor(Ytrain))

                validation_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(Xvalid), 
                                                                    torch.FloatTensor(Yvalid))

                valid_dataset = torch.FloatTensor(XtrainNNData['valid'])

            # Loader

                train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=batch,
                                                           shuffle=True,
                                                           num_workers=8)

                validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                                batch_size=batch,
                                                                shuffle=False, 
                                                                num_workers=8)

                valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                           batch_size=batch,
                                                           shuffle=False,
                                                           num_workers=8)

            # Model
                net = architecture

            # Loss function
                criterion = nn.BCEWithLogitsLoss()
                
            # Optimization algorithm
                optimizer = optim.RMSprop(net.parameters(), lr=learningRate, alpha=0.99, eps=1e-08, 
                                          weight_decay=0, momentum=0.9)

            # Training model
                trainNN(epoch, net, train_loader, optimizer, criterion, display_step=displayStep, 
                        valid_loader=validation_loader, use_GPU=useGPU)
                
            # Performance measuring
                validPrediction = predictNN(net, valid_loader, use_GPU=useGPU)
                score = log_loss(self.Ytrain['valid'], validPrediction)

                cvScore += score*(1/cvNumber)

                print("\nIntermediate score: %.5f" % score)

            print("\nValid log loss: %.5f\n" % cvScore)

        else:
            time = datetime.now()

        # Tensor
            train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(XtrainNNData[1]),
                                                           torch.FloatTensor(YtrainNNData[1]))

            tensor = {}
            for dataset in self.Xtrain: 
                tensor[dataset] = torch.FloatTensor(XtrainNNData[dataset])

        # Loader
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=batch,
                                                       shuffle=True,
                                                       num_workers=8)
            loader = {}
            for dataset in self.Xtrain: 
                loader[dataset] = torch.utils.data.DataLoader(dataset=tensor[dataset],
                                                              batch_size=batch,
                                                              shuffle=False,
                                                              num_workers=8)
        # Model
            net = architecture

        # Loss function
            criterion = nn.BCEWithLogitsLoss()
            
        # Optimization algorithm
            optimizer = optim.RMSprop(net.parameters(), lr=learningRate, alpha=0.99, eps=1e-08, 
                                      weight_decay=0, momentum=0.9)

        # Training model
            trainNN(epoch, net, train_loader, optimizer, criterion, display_step=displayStep, 
                    valid_loader=None, use_GPU=useGPU)

        # Predicting
            for dataset in self.Xtrain: 
                self.firstStagePrediction[dataset]['neuralNetwork'] = predictNN(net, loader[dataset], use_GPU=useGPU).flatten()

        del XtrainNNData, YtrainNNData

        print('\nRunning time {}'.format(diff(datetime.now(), time)))



    def _add_model(self, models): 

        """
        A model should be sklearn-friendly and have a "predict_proba" method
        """
        for name in models:
            self.modelNames.append(name)
            self.baggingSteps.append(models[name][1])
            self.nFeatures.append(models[name][2])
            self.stage.append(models[name][0])
            self.models.append(models[name][3])
            self.parameters.append(models[name][4])



    def training(self, nCores=-1, models = models, stageNumber=1, interaction=None):

    # Loading models
        self._add_model(models)

    # Defining score

        score = make_scorer(score_func = log_loss)
        time = datetime.now()
        
    # First stage

        print('\n---------------------------------------------')
        print('>> Processing first stage')

        features = [name for name in self.Xtrain[1].columns]
        time1 = datetime.now()

        if self.notYetNN_train:
            firstStagePrediction = {}
            for dataset in self.Xtrain:
                firstStagePrediction[dataset] = pd.DataFrame()
        else:
            firstStagePrediction = self.firstStagePrediction
            del self.firstStagePrediction

        for name, model, parameters, baggingSteps, nFeatures, stage in zip(self.modelNames, self.models, self.parameters, self.baggingSteps, self.nFeatures, self.stage):
            
            if stage == 1:
                print('\n>> Processing {} ------\n'.format(name))

                for step in range(baggingSteps):
        
                    time2 = datetime.now()
                    print("Step {}".format(step+1), end = '...')

                # Creating data
                    np.random.shuffle(features)

                    inter = {}
                    for dataset in self.Xtrain:
                        inter[dataset] = deepcopy(self.Xtrain[dataset][features[:nFeatures]])

                # Kmeans trick
                    if 1 in self.kmeanStage:
                        if self.kmeansInteraction:
                            for dataset in self.Xtrain:
                                for feature in inter[dataset].columns:
                                    for meta in self.kmeanDist[dataset].columns:
                                        inter[dataset]['{}_{}'.format(feature, meta)] = inter[dataset][feature].values * self.kmeanDist[dataset][meta].values
                        else:
                            for dataset in self.Xtrain:
                                inter[dataset] = pd.concat([inter[dataset].reset_index(drop=True), 
                                                            self.kmeanDist[dataset].reset_index(drop=True)], axis=1)
                    else:
                        pass

                # Kernel PCA
                    if 1 in self.pcaStage:
                        if self.pcaInteraction:
                            for dataset in self.Xtrain:
                                for feature in inter[dataset].columns:
                                    for meta in self.KernelPCA[dataset].columns:
                                        inter[dataset]['{}_{}'.format(feature, meta)] = inter[dataset][feature].values * self.KernelPCA[dataset][meta].values
                        else:
                            for dataset in self.Xtrain:
                                inter[dataset] = pd.concat([inter[dataset].reset_index(drop=True), 
                                                            self.KernelPCA[dataset].reset_index(drop=True)], axis=1)
                    else:
                        pass

                # Keeping only processed features
                    if False:
                        for dataset in self.Xtrain:
                            inter[dataset].drop([feature], inplace=True, axis=1)

                # Tuning
                    gscv = GridSearchCV(model, parameters, scoring=score, n_jobs=nCores, cv=4)
                    gscv.fit(inter[1], self.Ytrain[1])                                                            ## FIRST STAGE TRAINING ON XTRAIN1

                # Saving best predictions
                    for dataset in self.Xtrain:
                        firstStagePrediction[dataset]['{}_prediction_{}'.format(name, step+1)] = gscv.predict_proba(inter[dataset])[:,1]

                    del inter

                    print('done in {}'.format(diff(datetime.now(), time2)))
                    if self.evaluate:
                        print('log loss : %.5f\n' %
                            (log_loss(self.Ytrain['test'], firstStagePrediction['test']['{}_prediction_{}'.format(name,step+1)])))

        del self.Xtrain[1], self.Ytrain[1], firstStagePrediction[1]

        print('\nFirst stage running time {}'.format(diff(datetime.now(), time1)))

    # Second stage

        if stageNumber == 2:

            print('\n\n---------------------------------------------')
            print('>> Processing second stage')

            features = [name for name in firstStagePrediction[1].columns]
            time1 = datetime.now()

            secondStagePrediction = {}
            for dataset in self.Xtrain:
                secondStagePrediction[dataset] = pd.DataFrame()

            for name, model, parameters, baggingSteps, nFeatures, stage in zip(self.modelNames, self.models, self.parameters, self.baggingSteps, self.nFeatures, self.stage):
            
                if stage == 2:
                    print('\n>> Processing {} ------\n'.format(name))

                    for step in range(baggingSteps):
            
                        time2 = datetime.now()
                        print("Step {}".format(step+1), end = '...')

                    # Creating data
                        np.random.shuffle(features)

                        inter = {}
                        for dataset in self.Xtrain:
                            inter[dataset] = deepcopy(firstStagePrediction[dataset][features[:nFeatures]])

                    # Kmeans trick
                        if 1 in self.kmeanStage:
                            if self.kmeansInteraction:
                                for dataset in self.Xtrain:
                                    for feature in inter[dataset].columns:
                                        for meta in self.kmeanDist[dataset].columns:
                                            inter[dataset]['{}_{}'.format(feature, meta)] = inter[dataset][feature].values * self.kmeanDist[dataset][meta].values
                            else:
                                for dataset in self.Xtrain:
                                    inter[dataset] = pd.concat([inter[dataset].reset_index(drop=True), 
                                                                self.kmeanDist[dataset].reset_index(drop=True)], axis=1)
                        else:
                            pass

                    # Kernel PCA
                        if 1 in self.pcaStage:
                            if self.pcaInteraction:
                                for dataset in self.Xtrain:
                                    for feature in inter[dataset].columns:
                                        for meta in self.KernelPCA[dataset].columns:
                                            inter[dataset]['{}_{}'.format(feature, meta)] = inter[dataset][feature].values * self.KernelPCA[dataset][meta].values
                            else:
                                for dataset in self.Xtrain:
                                    inter[dataset] = pd.concat([inter[dataset].reset_index(drop=True), 
                                                                self.KernelPCA[dataset].reset_index(drop=True)], axis=1)
                        else:
                            pass

                    # Keeping only processed features
                        if False:
                            for dataset in self.Xtrain:
                                inter[dataset].drop([feature], inplace=True, axis=1)

                    # Tuning
                        gscv = GridSearchCV(model, parameters, scoring=score, n_jobs=nCores, cv=5)
                        gscv.fit(inter[2], self.Ytrain[2])                                                        ## SECOND STAGE TRAINING ON FIRST STAGE PREDICTION OF XTRAIN2
                                                                                                                  
                    # Saving best predictions
                        for dataset in self.Xtrain:
                            secondStagePrediction[dataset]['{}_prediction_{}'.format(name, step+1)] = gscv.predict_proba(inter[dataset])[:,1]

                        del inter

                        print('done in {}'.format(diff(datetime.now(), time2)))
                        if self.evaluate:
                            print('log loss : %.5f\n' %
                                (log_loss(self.Ytrain['test'], secondStagePrediction['test']['{}_prediction_{}'.format(name,step+1)])))

            del self.Xtrain[2], self.Ytrain[2], firstStagePrediction, secondStagePrediction[1], secondStagePrediction[2]

            print('Second stage running time {}'.format(diff(datetime.now(), time1)))

    # For compilation

        if stageNumber == 1:
            self.compilation_data = firstStagePrediction
            self.datasetToUse = 2
        elif stageNumber == 2:
            self.compilation_data = secondStagePrediction
            self.datasetToUse = 3

    # Kmeans trick
        if self.datasetToUse in self.kmeanStage:
            if self.kmeansInteraction:
                for dataset in self.Xtrain:
                    for feature in self.compilation_data[dataset].columns:
                        for meta in self.kmeanDist[dataset].columns:
                            self.compilation_data[dataset]['{}_{}'.format(feature, meta)] = self.compilation_data[dataset][feature].values * self.kmeanDist[dataset][meta].values
            else:
                for dataset in self.Xtrain:
                    self.compilation_data[dataset] = pd.concat([self.compilation_data[dataset].reset_index(drop=True), 
                                                                self.kmeanDist[dataset].reset_index(drop=True)], axis=1)
        else:
            pass

    # Kernel PCA
        if self.datasetToUse in self.pcaStage:
            if self.pcaInteraction:
                for dataset in self.compilation_data:
                    for feature in self.compilation_data[dataset].columns:
                        for meta in self.KernelPCA[dataset].columns:
                            self.compilation_data[dataset]['{}_{}'.format(feature, meta)] = self.compilation_data[dataset][feature].values * self.KernelPCA[dataset][meta].values
            else:
                for dataset in self.Xtrain:
                   self.compilation_data[dataset] = pd.concat([self.compilation_data[dataset].reset_index(drop=True), 
                                                                self.KernelPCA[dataset].reset_index(drop=True)], axis=1)
        else:
            pass


#----------------------------------------------------------------------------------------------
#---------------------------------------- COMPILATION -----------------------------------------


    def compile(self, nCores=-1, neuralNetworkCompiler=False, architecture=None, learningRate=0.0001, 
                batch=64, epoch=2, cvNumber=1, displayStep=10000, useGPU=False):

    # Defining score

        score = make_scorer(score_func = log_loss)
        time = datetime.now()
        
    # Data
        self.finalPrediction = {}
        for dataset in self.Xtrain:
            self.finalPrediction[dataset] = pd.DataFrame()

        if neuralNetworkCompiler:
            
            print('\n---------------------------------------------')
            print('>> Processing compilation [Neural Network]\n')

            if self.notYetNN_comp:
                for dataset in self.Xtrain:
                    self.compilation_data[dataset] = np.array(self.compilation_data[dataset])
                    if dataset != 'real_data':
                        self.Ytrain[dataset] = np.array(self.Ytrain[dataset].values)
                        self.Ytrain[dataset] = self.Ytrain[dataset].reshape(self.Ytrain[dataset].shape[0], 1)

                self.compilation_data[self.datasetToUse] = np.concatenate((self.compilation_data[self.datasetToUse], self.compilation_data['test']), axis=0)
                self.Ytrain[self.datasetToUse] = np.concatenate((self.Ytrain[self.datasetToUse], self.Ytrain['test']), axis=0)

                del self.compilation_data['test']
                self.notYetNN_comp = False

            if self.evaluate:
            # Tuning hyperparameter
                cvScore = 0

                time = datetime.now()

                for i in range(cvNumber):
                    print('\nLoop number {}'.format(i+1))

                    Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(self.compilation_data[self.datasetToUse], self.Ytrain[self.datasetToUse], test_size=0.25)

                # Tensor

                    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(Xtrain), 
                                                                   torch.FloatTensor(Ytrain))

                    validation_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(Xvalid), 
                                                                        torch.FloatTensor(Yvalid))

                    valid_dataset = torch.FloatTensor(self.compilation_data['valid'])

                # Loader

                    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                               batch_size=batch,
                                                               shuffle=True,
                                                               num_workers=8)

                    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                                    batch_size=batch,
                                                                    shuffle=False, 
                                                                    num_workers=8)

                    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                               batch_size=batch,
                                                               shuffle=False,
                                                               num_workers=8)
                # Model
                    net = architecture

                # Loss function
                    criterion = nn.BCEWithLogitsLoss()
                    
                # Optimization algorithm
                    optimizer = optim.RMSprop(net.parameters(), lr=learningRate, alpha=0.99, eps=1e-08, 
                                              weight_decay=0, momentum=0.9)

                # Training model
                    trainNN(epoch, net, train_loader, optimizer, criterion, display_step=displayStep, 
                            valid_loader=validation_loader, use_GPU=useGPU)

                # Predicting
                    validPrediction = predictNN(net, valid_loader, use_GPU=useGPU)
                    
                # Performance measuring
                    score = log_loss(self.Ytrain['valid'], validPrediction)

                    cvScore += score*(1/cvNumber)

                    print("\nIntermediate score: %.5f" % score)

                print("\nFinal valid log loss: %.5f\n" % cvScore)
                print('\nTotal running time {}'.format(diff(datetime.now(), time)))

            else:
                time = datetime.now()

            # Tensor
                train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(self.compilation_data[self.datasetToUse]), 
                                                               torch.FloatTensor(self.Ytrain[self.datasetToUse]))

                real_dataset = torch.FloatTensor(self.compilation_data['real_data'])

            # Loader
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=batch,
                                                           shuffle=True,
                                                           num_workers=8)

                real_loader = torch.utils.data.DataLoader(dataset=real_dataset,
                                                          batch_size=batch,
                                                          shuffle=False,
                                                          num_workers=8)
            # Model
                net = architecture

            # Loss function
                criterion = nn.BCEWithLogitsLoss()
                
            # Optimization algorithm
                optimizer = optim.RMSprop(net.parameters(), lr=learningRate, alpha=0.99, eps=1e-08, 
                                          weight_decay=0, momentum=0.9)

            # Training model
                trainNN(epoch, net, train_loader, optimizer, criterion, display_step=displayStep, 
                        valid_loader=None, use_GPU=useGPU)

            # Predicting
                self.finalPrediction['real_data'] = predictNN(net, real_loader, use_GPU=useGPU)

                print('\nRunning time {}'.format(diff(datetime.now(), time)))
        else:
            # Tuning
            for name, model, parameters, baggingSteps, nFeatures, stage in zip(self.modelNames, self.models, self.parameters, self.baggingSteps, self.nFeatures, self.stage):
                
                if stage == self.datasetToUse:                                                                                     ## There is normally only one model that fits here !!!!

                    print('\n---------------------------------------------')
                    print('>> Processing compilation [{}]\n'.format(name))

                    gscv = GridSearchCV(model, parameters, scoring=score, n_jobs=nCores, cv=5)
                    gscv.fit(self.compilation_data[self.datasetToUse], self.Ytrain[self.datasetToUse])                        ## COMPILATION TRAINING ON SECOND STAGE PREDICTION OF XTRAIN3
                                                                                                                              ## IF THERE IS TWO STAGES, ELSE ON XTRAIN2
                    # Final prediction
                    for dataset in self.Xtrain:
                        self.finalPrediction[dataset]['final_prediction'] = gscv.predict_proba(self.compilation_data[dataset])[:,1]

            print('\nCompilation running time {}'.format(diff(datetime.now(), time)))
            if self.evaluate:
                print('\nFinal test log loss : %.5f' %
                    (log_loss(self.Ytrain['test'], self.finalPrediction['test']['final_prediction'])))                        
                print('Final valid log loss : %.5f\n' %
                    (log_loss(self.Ytrain['valid'], self.finalPrediction['valid']['final_prediction'])))


#----------------------------------------------------------------------------------------------
#---------------------------------------- SUBMISSION ------------------------------------------


    def submit(self, submissionNumber, week):

        if not self.evaluate:

            submit = pd.DataFrame()
            submit['id'] = self.ids
            submit['probability_{}'.format(self.type)] = self.finalPrediction['real_data']

        # Saving prediction
            submit.to_csv('../../../Datasets/Numerai/w{0}/submission{1}_{2}.csv'.format(week, submissionNumber, self.type), index = False)

        # Automated submission through the numerai API
            pass


