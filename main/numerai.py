





"""
Main class designed to quickly evaluate different model architectures over Numerai dataset



Next steps:
    - Add knn prediction from R or redundant ?
    - Add a stageNumber = 0 option
    - NMF (Non-negative matrix factorization) instead of PCA: assumed to be better for tree-based models
    - Feature interaction: compute all interaction and then select more important by fitting a randomForest !
    - Add noise for autoencoder input (prevent from overfitting)
    - Memory optimization (inter & feature only when computing)
    - Add more randomness (random feature engineering ?)
    - Using era (cv only by era, add era info on all rows)
    - hardcore EDA



Author: Hugo Perrin
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
from sklearn.decomposition import PCA
from sklearn import cluster

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Deep learning
sys.path.append('../pytorch/')
from utils import trainNN, predictNN
from utils import train_autoencoder, predict_autoencoder, get_encoded
import torch.utils.data as utils
import torch.optim as optim
import torch.nn as nn
import torch

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

        self.kmeanStage = []
        self.knnStage = []
        self.pcaStage = []
        self.autoencoderStage = []

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



    def load_data(self, stageNumber, evaluate, knn=False):

        print('\n---------------------------------------------')
        print('>> Loading data', end='...')
        Xtrain = pd.read_csv("../../../Datasets/Numerai/w{}/numerai_training_data.csv".format(self.week))
        Xvalid = pd.read_csv("../../../Datasets/Numerai/w{}/numerai_tournament_data.csv".format(self.week))

        self.evaluate = evaluate
        self.stageNumber = stageNumber

        real_data = Xvalid.copy(True)
        self.ids = Xvalid['id']

        Xvalid = Xvalid[Xvalid['data_type'] == 'validation']

        Ytrain = deepcopy(Xtrain['target_{}'.format(self.type)])
        Yvalid = Xvalid['target_{}'.format(self.type)]

        Xtrain.drop(['id', 'era', 'data_type', 'target_bernie', 'target_charles', 'target_elizabeth', 'target_jordan', 'target_ken'], inplace=True, axis=1)
        Xvalid.drop(['id', 'era', 'data_type', 'target_bernie', 'target_charles', 'target_elizabeth', 'target_jordan', 'target_ken'], inplace=True, axis=1)
        real_data.drop(['id', 'era', 'data_type', 'target_bernie', 'target_charles', 'target_elizabeth', 'target_jordan', 'target_ken'], inplace=True, axis=1)

        # If using knn be carefull not have any of the training points in the test dataset
        if knn:
            ids = pd.read_csv("../../../Datasets/Numerai/w{0}/train_ids_{1}.csv".format(self.week, self.type))

            # Saving 
            knnXtrainingPoints = Xtrain.iloc[ids['x'].values]
            knnYtrainingPoints = Ytrain.iloc[ids['x'].values]

            # Droping
            Xtrain.drop(ids['x'].values, inplace=True, axis=0)
            Ytrain.drop(ids['x'].values, inplace=True, axis=0)

            # Adapting proportion
            prop1 = 0.6*(ids.shape[0]+Xtrain.shape[0])/Xtrain.shape[0]
            prop2 = 0.5*(ids.shape[0]+Xtrain.shape[0])/Xtrain.shape[0]
            prop3 = 0.72*(ids.shape[0]+Xtrain.shape[0])/Xtrain.shape[0]
            prop4 = 0.67*(ids.shape[0]+Xtrain.shape[0])/Xtrain.shape[0]

            del ids

        else:
            prop1 = 0.6
            prop2 = 0.5
            prop3 = 0.72
            prop4 = 0.67


        if stageNumber == 1:
            if self.evaluate:
                # Xtrain1 = 40%, Xtrain2 = 40%, Xtest = 20%
                Xtrain1, Xtrain2, Ytrain1, Ytrain2 = train_test_split(Xtrain, Ytrain, test_size=prop1)
                Xtrain2, Xtest, Ytrain2, Ytest = train_test_split(Xtrain2, Ytrain2, test_size=0.33)

                # Adding KNN training data from R (to avoid overfitting by having them all in 1 and not in test)
                if knn:
                    Xtrain1 = pd.concat([Xtrain1,
                                         knnXtrainingPoints], axis=0)
                    Ytrain1 = pd.concat([Ytrain1,
                                         knnYtrainingPoints], axis=0)

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
                Xtrain1, Xtrain2, Ytrain1, Ytrain2 = train_test_split(Xtrain, Ytrain, test_size=prop2)

                # Adding KNN training data from R (to avoid overfitting by having them all in 1 and not in test)
                if knn:
                    Xtrain1 = pd.concat([Xtrain1,
                                         knnXtrainingPoints], axis=0)
                    Ytrain1 = pd.concat([Ytrain1,
                                         knnYtrainingPoints], axis=0)

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
                # Xtrain1 = 28%, Xtrain2 = 28%, Xtrain3 = 28%, Xtest = 16%
                Xtrain1, Xtrain2, Ytrain1, Ytrain2 = train_test_split(Xtrain, Ytrain, test_size=prop3)
                Xtrain2, Xtrain3, Ytrain2, Ytrain3 = train_test_split(Xtrain2, Ytrain2, test_size=0.61)
                Xtrain3, Xtest, Ytrain3, Ytest = train_test_split(Xtrain3, Ytrain3, test_size=0.365)

                # Adding KNN training data from R (to avoid overfitting by having them all in 1 and not in test)
                if knn:
                    Xtrain1 = pd.concat([Xtrain1,
                                         knnXtrainingPoints], axis=0)
                    Ytrain1 = pd.concat([Ytrain1,
                                         knnYtrainingPoints], axis=0)

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
                Xtrain1, Xtrain2, Ytrain1, Ytrain2 = train_test_split(Xtrain, Ytrain, test_size=prop4)
                Xtrain2, Xtrain3, Ytrain2, Ytrain3 = train_test_split(Xtrain2, Ytrain2, test_size=0.5)

                # Adding KNN training data from R (to avoid overfitting by having them all in 1 and not in test)
                if knn:
                    Xtrain1 = pd.concat([Xtrain1,
                                         knnXtrainingPoints], axis=0)
                    Ytrain1 = pd.concat([Ytrain1,
                                         knnYtrainingPoints], axis=0)

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



    def kmeansTrick(self, k, stage, interaction):

        print('\n---------------------------------------------')
        print('>> Processing Kmeans ------\n')

        self.kmeanStage = stage
        self.kmeansInteraction = interaction

    # Unsupervised learning
        print('Fitting model', end='...')
        model = cluster.KMeans(n_clusters=k, precompute_distances=False, n_jobs=-1)
        model.fit(self.Xtrain[1])
        print('done')

    # Feature engineering
        print('Generating kmeans features', end='...')
        self.kmeanDist = {}
        for dataset in self.Xtrain:
            self.kmeanDist[dataset] = model.transform(self.Xtrain[dataset])
            self.kmeanDist[dataset] = pd.DataFrame(self.kmeanDist[dataset], columns=['kmeans{}'.format(i) for i in range(k)])
        print('done')



    def knnDistances(self, name, stage, interaction):
        """
        Load data computed with R script and assign it to the good dataset
        """

        print('\n---------------------------------------------')
        print('>> Processing KNN ------\n')

        self.knnStage = stage
        self.knnInteraction = interaction

        tournament = pd.read_csv("/home/hugoperrin/Bureau/Datasets/Numerai/w{}/knnFeatures_tournament_{}.csv".format(self.week, name))
        train = pd.read_csv("/home/hugoperrin/Bureau/Datasets/Numerai/w{}/knnFeatures_train_{}.csv".format(self.week, name))

        self.knnDistances = {}

        print('Generating knn features', end='...')

        for dataset in self.Xtrain:
            if dataset in ['real_data', 'valid']:
                self.knnDistances[dataset] = pd.DataFrame(index=self.Xtrain[dataset].index)
                self.knnDistances[dataset] = pd.merge(self.knnDistances[dataset], tournament, left_index=True, right_index=True)
            else:
                self.knnDistances[dataset] = pd.DataFrame(index=self.Xtrain[dataset].index)
                self.knnDistances[dataset] = pd.merge(self.knnDistances[dataset], train, left_index=True, right_index=True)
        print('done')



    def PCA(self, n_components, stage, interaction):

        print('\n---------------------------------------------')
        print('>> Processing PCA ------\n')

        self.pcaStage = stage
        self.pcaInteraction = interaction

    # Unsupervised learning
        print('Fitting model', end='...')
        model = PCA(n_components=n_components)
        model.fit(self.Xtrain[1])
        print('done')

    # Feature engineering
        print('Generating PCA features', end='...')
        self.PCA = {}
        for dataset in self.Xtrain:
            self.PCA[dataset] = model.transform(self.Xtrain[dataset])
            self.PCA[dataset] = pd.DataFrame(self.PCA[dataset], columns=['PCA{}'.format(i) for i in range(n_components)])
        print('done')



    def autoEncoder(self, stage, interaction, 
                    layers, dropout, learningRate=0.0001, batch=64, epoch=5, 
                    cvNumber=1, displayStep=1000, useGPU=True, evaluate=True):

        print('\n---------------------------------------------')
        print('>> Processing AutoEncoder ------')

        self.autoencoderStage = stage
        self.autoencoderInteraction = interaction

    # Model depth
        if len(layers) == 3:

            encoding_size = layers[1]

            class Autoencoder(nn.Module):

                def __init__(self):
                    super(Autoencoder, self).__init__()

                    self.encoder = nn.Sequential(
                                            nn.Linear(50, layers[0]),
                                            nn.Tanh(),
                                            nn.Dropout(dropout),
                                            nn.Linear(layers[0], encoding_size))

                    self.decoder = nn.Sequential(
                                            nn.Tanh(),
                                            nn.Dropout(dropout),
                                            nn.Linear(encoding_size, layers[2]),
                                            nn.Tanh(),
                                            nn.Dropout(dropout),
                                            nn.Linear(layers[2], 50))

                def forward(self, x):
                    
                    encoded = self.encoder(x)
                    out = self.decoder(encoded)

                    return encoded, out

        elif len(layers) == 1:

            encoding_size = layers[0]

            class Autoencoder(nn.Module):

                def __init__(self):
                    super(Autoencoder, self).__init__()

                    self.encoder = nn.Sequential(
                                            nn.Linear(50, encoding_size))

                    self.decoder = nn.Sequential(
                                            nn.Tanh(),
                                            nn.Dropout(dropout),
                                            nn.Linear(encoding_size, 50))

                def forward(self, x):
                    
                    encoded = self.encoder(x)
                    out = self.decoder(encoded)

                    return encoded, out

    # Data
        XtrainNNData = {}
        for dataset in self.Xtrain:
            if evaluate & (dataset in [2,3]):
                pass  # For training only 1, test and valid are needed
            else:
                XtrainNNData[dataset] = deepcopy(self.Xtrain[dataset])
                XtrainNNData[dataset] = np.array(XtrainNNData[dataset])

        if self.evaluate:
            XtrainNNData['nn'] = np.concatenate((XtrainNNData[1], XtrainNNData['test']), axis=0)
        else:
            pass # Normally that never happens (self.evaluate == False & evaluate == True) so no use of defining another 'nn' dataset

    # Tuning hyperparameter

        if evaluate:

        # Optimizing memory usage
            del XtrainNNData[1], XtrainNNData['test']

            cvScore = 0

            time = datetime.now()

            for i in range(cvNumber):
                print('\nLoop number {}'.format(i+1))

                Xtrain, Xvalid = train_test_split(XtrainNNData['nn'], test_size=0.25)  # Input == target 

            # Tensor

                train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(Xtrain), 
                                                               torch.FloatTensor(Xtrain))

                validation_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(Xvalid), 
                                                                    torch.FloatTensor(Xvalid))

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
                net = Autoencoder()

            # Loss function
                criterion = nn.MSELoss()
                
            # Optimization algorithm
                optimizer = optim.RMSprop(net.parameters(), lr=learningRate, alpha=0.99, eps=1e-08, 
                                          weight_decay=0, momentum=0.9)

            # Training model
                train_autoencoder(epoch, net, train_loader, optimizer, criterion, display_step=displayStep, 
                                  valid_loader=validation_loader, use_GPU=useGPU)
                
            # Performance measuring
                validPrediction = pd.DataFrame(predict_autoencoder(net, valid_loader, use_GPU=useGPU))
                score = mean_squared_error(XtrainNNData['valid'], validPrediction)

                cvScore += score*(1/cvNumber)

            print("\nValid log loss: %.5f\n" % cvScore)

    # Getting encoding

        else:

            time = datetime.now()

        # Tensor
            train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(XtrainNNData[1]),
                                                           torch.FloatTensor(XtrainNNData[1]))

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
            net = Autoencoder()

        # Loss function
            criterion = nn.MSELoss()
            
        # Optimization algorithm
            optimizer = optim.RMSprop(net.parameters(), lr=learningRate, alpha=0.99, eps=1e-08, 
                                      weight_decay=0, momentum=0.9)

        # Training model
            train_autoencoder(epoch, net, train_loader, optimizer, criterion, display_step=displayStep, 
                              valid_loader=None, use_GPU=useGPU)

        # Predicting
            print('\nGenerating encodied features', end='...')
            self.autoencoderFeature = {}
            for dataset in self.Xtrain:
                self.autoencoderFeature[dataset] = get_encoded(net, loader[dataset], use_GPU=useGPU)
                self.autoencoderFeature[dataset] = pd.DataFrame(self.autoencoderFeature[dataset], columns=['encoded{}'.format(i) for i in range(encoding_size)])
            print('done\n')

        del XtrainNNData

        print('Running time {}\n'.format(diff(datetime.now(), time)))



#----------------------------------------------------------------------------------------------
#------------------------------------------- MODELS -------------------------------------------



    def trainingNN(self, layers, dropout, learningRate=0.0001, batch=64, epoch=5, 
                   cvNumber=1, displayStep=1000, useGPU=True, evaluate=True):

        self.notYetNN_train = False

    # Model
        class NN(nn.Module):

            def __init__(self):
                super(NN, self).__init__()

                self.linear = nn.Sequential(
                    nn.Linear(layers[0], layers[1]),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(layers[1], 1))

            def forward(self, x):
                out = self.linear(x)
                return out
        
    # Data
        self.firstStagePrediction = {}
        for dataset in self.Xtrain:
            self.firstStagePrediction[dataset] = pd.DataFrame()
            
        print('\n---------------------------------------------')
        print('>> Processing Neural Network ------')

        XtrainNNData = {}
        YtrainNNData = {}
        for dataset in self.Xtrain:

            if evaluate & (dataset in [2,3]):
                pass  # For training only 1, test and valid are needed
            else:

                XtrainNNData[dataset] = deepcopy(self.Xtrain[dataset])

            # Kmeans trick ------------ 

                if 1 in self.kmeanStage:
                    if self.kmeansInteraction:
                        for feature in self.Xtrain[dataset].columns:
                            for meta in self.kmeanDist[dataset].columns:
                                XtrainNNData[dataset]['{}_{}'.format(feature, meta)] = self.Xtrain[dataset][feature].values * self.kmeanDist[dataset][meta].values
                    else:
                        XtrainNNData[dataset] = pd.concat([XtrainNNData[dataset].reset_index(drop=True),
                                                           self.kmeanDist[dataset].reset_index(drop=True)], axis=1)

            # Kernel PCA --------------

                if 1 in self.pcaStage:
                    if self.pcaInteraction:
                        for feature in self.Xtrain[dataset].columns:
                            for meta in self.PCA[dataset].columns:
                                XtrainNNData[dataset]['{}_{}'.format(feature, meta)] = self.Xtrain[dataset][feature].values * self.PCA[dataset][meta].values
                    else:
                        XtrainNNData[dataset] = pd.concat([XtrainNNData[dataset].reset_index(drop=True),
                                                           self.PCA[dataset].reset_index(drop=True)], axis=1)

            # Autoencoding -------------

                if 1 in self.autoencoderStage:
                    if self.autoencoderInteraction:
                        for feature in self.Xtrain[dataset].columns:
                            for meta in self.autoencoderFeature[dataset].columns:
                                XtrainNNData[dataset]['{}_{}'.format(feature, meta)] = self.Xtrain[dataset][feature].values * self.autoencoderFeature[dataset][meta].values
                    else:
                        XtrainNNData[dataset] = pd.concat([XtrainNNData[dataset].reset_index(drop=True),
                                                           self.autoencoderFeature[dataset].reset_index(drop=True)], axis=1)

            # Knn distances ------------

                if 1 in self.knnStage:
                    if self.knnInteraction:
                        for feature in self.Xtrain[dataset].columns:
                            for meta in self.knnDistances[dataset].columns:
                                XtrainNNData[dataset]['{}_{}'.format(feature, meta)] = self.Xtrain[dataset][feature].values * self.knnDistances[dataset][meta].values
                    else:
                        XtrainNNData[dataset] = pd.concat([XtrainNNData[dataset].reset_index(drop=True),
                                                           self.knnDistances[dataset].reset_index(drop=True)], axis=1)

            # To array 
                XtrainNNData[dataset] = np.array(XtrainNNData[dataset])

                if dataset != 'real_data':
                    YtrainNNData[dataset] = np.array(self.Ytrain[dataset].values)
                    YtrainNNData[dataset] = YtrainNNData[dataset].reshape(YtrainNNData[dataset].shape[0], 1)

        if self.evaluate:
            XtrainNNData['nn'] = np.concatenate((XtrainNNData[1], XtrainNNData['test']), axis=0)
            YtrainNNData['nn'] = np.concatenate((YtrainNNData[1], YtrainNNData['test']), axis=0)
        else:
            pass # Normally that never happens (self.evaluate == False & evaluate == True) so no use of defining another 'nn' dataset

    # Tuning hyperparameter

        if evaluate:

            # Optimizing memory usage
            del XtrainNNData[1], XtrainNNData['test'], YtrainNNData[1], YtrainNNData['test']

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
                net = NN()

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
                score = log_loss(YtrainNNData['valid'], validPrediction)

                cvScore += score*(1/cvNumber)

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
            net = NN()

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

        print('Running time {}\n'.format(diff(datetime.now(), time)))



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



    def training(self, nCores=-1, models = models):

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

                    columns = inter[1].columns

                # Kmeans trick
                    if 1 in self.kmeanStage:
                        if self.kmeansInteraction:
                            for dataset in self.Xtrain:
                                for feature in columns:
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
                                for feature in columns:
                                    for meta in self.PCA[dataset].columns:
                                        inter[dataset]['{}_{}'.format(feature, meta)] = inter[dataset][feature].values * self.PCA[dataset][meta].values
                        else:
                            for dataset in self.Xtrain:
                                inter[dataset] = pd.concat([inter[dataset].reset_index(drop=True), 
                                                            self.PCA[dataset].reset_index(drop=True)], axis=1)
                    else:
                        pass

                # Autoencoder
                    if 1 in self.autoencoderStage:
                        if self.autoencoderInteraction:
                            for dataset in self.Xtrain:
                                for feature in columns:
                                    for meta in self.autoencoderFeature[dataset].columns:
                                        inter[dataset]['{}_{}'.format(feature, meta)] = inter[dataset][feature].values * self.autoencoderFeature[dataset][meta].values
                        else:
                            for dataset in self.Xtrain:
                                inter[dataset] = pd.concat([inter[dataset].reset_index(drop=True), 
                                                            self.autoencoderFeature[dataset].reset_index(drop=True)], axis=1)
                    else:
                        pass

                # Knn distances
                    if 1 in self.knnStage:
                        if self.knnInteraction:
                            for dataset in self.Xtrain:
                                for feature in columns:
                                    for meta in self.knnDistances[dataset].columns:
                                        inter[dataset]['{}_{}'.format(feature, meta)] = inter[dataset][feature].values * self.knnDistances[dataset][meta].values
                        else:
                            for dataset in self.Xtrain:
                                inter[dataset] = pd.concat([inter[dataset].reset_index(drop=True), 
                                                            self.knnDistances[dataset].reset_index(drop=True)], axis=1)
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

        if self.stageNumber == 2:

            print('\n\n---------------------------------------------')
            print('>> Processing second stage')

            features = [name for name in firstStagePrediction[2].columns]
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

                        columns = inter[2].columns

                    # Kmeans trick
                        if 2 in self.kmeanStage:
                            if self.kmeansInteraction:
                                for dataset in self.Xtrain:
                                    for feature in columns:
                                        for meta in self.kmeanDist[dataset].columns:
                                            inter[dataset]['{}_{}'.format(feature, meta)] = inter[dataset][feature].values * self.kmeanDist[dataset][meta].values
                            else:
                                for dataset in self.Xtrain:
                                    inter[dataset] = pd.concat([inter[dataset].reset_index(drop=True), 
                                                                self.kmeanDist[dataset].reset_index(drop=True)], axis=1)
                        else:
                            pass

                    # Kernel PCA
                        if 2 in self.pcaStage:
                            if self.pcaInteraction:
                                for dataset in self.Xtrain:
                                    for feature in columns:
                                        for meta in self.PCA[dataset].columns:
                                            inter[dataset]['{}_{}'.format(feature, meta)] = inter[dataset][feature].values * self.PCA[dataset][meta].values
                            else:
                                for dataset in self.Xtrain:
                                    inter[dataset] = pd.concat([inter[dataset].reset_index(drop=True), 
                                                                self.PCA[dataset].reset_index(drop=True)], axis=1)
                        else:
                            pass

                    # Autoencoder
                        if 2 in self.autoencoderStage:
                            if self.autoencoderInteraction:
                                for dataset in self.Xtrain:
                                    for feature in columns:
                                        for meta in self.autoencoderFeature[dataset].columns:
                                            inter[dataset]['{}_{}'.format(feature, meta)] = inter[dataset][feature].values * self.autoencoderFeature[dataset][meta].values
                            else:
                                for dataset in self.Xtrain:
                                    inter[dataset] = pd.concat([inter[dataset].reset_index(drop=True), 
                                                                self.autoencoderFeature[dataset].reset_index(drop=True)], axis=1)
                        else:
                            pass

                    # Knn distances
                        if 2 in self.knnStage:
                            if self.knnInteraction:
                                for dataset in self.Xtrain:
                                    for feature in columns:
                                        for meta in self.knnDistances[dataset].columns:
                                            inter[dataset]['{}_{}'.format(feature, meta)] = inter[dataset][feature].values * self.knnDistances[dataset][meta].values
                            else:
                                for dataset in self.Xtrain:
                                    inter[dataset] = pd.concat([inter[dataset].reset_index(drop=True), 
                                                                self.knnDistances[dataset].reset_index(drop=True)], axis=1)
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

            del self.Xtrain[2], self.Ytrain[2], firstStagePrediction, secondStagePrediction[2]

            print('Second stage running time {}'.format(diff(datetime.now(), time1)))

    # Memory efficiency
        del self.Xtrain

    # For compilation
        if self.stageNumber == 1:
            self.compilation_data = firstStagePrediction
            del firstStagePrediction
            self.datasetToUse = 2
        elif self.stageNumber == 2:
            self.compilation_data = secondStagePrediction
            del secondStagePrediction
            self.datasetToUse = 3

        columns = self.compilation_data[self.datasetToUse].columns

    # Kmeans trick
        if self.datasetToUse in self.kmeanStage:
            if self.kmeansInteraction:
                for dataset in self.compilation_data:
                    for feature in columns:
                        for meta in self.kmeanDist[dataset].columns:
                            self.compilation_data[dataset]['{}_{}'.format(feature, meta)] = self.compilation_data[dataset][feature].values * self.kmeanDist[dataset][meta].values
            else:
                for dataset in self.compilation_data:
                    self.compilation_data[dataset] = pd.concat([self.compilation_data[dataset].reset_index(drop=True), 
                                                                self.kmeanDist[dataset].reset_index(drop=True)], axis=1)
        else:
            pass

    # Memory efficiency
        try:
            del self.kmeanDist
        except:
            pass

    # Kernel PCA
        if self.datasetToUse in self.pcaStage:
            if self.pcaInteraction:
                for dataset in self.compilation_data:
                    for feature in columns:
                        for meta in self.PCA[dataset].columns:
                            self.compilation_data[dataset]['{}_{}'.format(feature, meta)] = self.compilation_data[dataset][feature].values * self.PCA[dataset][meta].values
            else:
                for dataset in self.compilation_data:
                    self.compilation_data[dataset] = pd.concat([self.compilation_data[dataset].reset_index(drop=True), 
                                                                self.PCA[dataset].reset_index(drop=True)], axis=1)
        else:
            pass

    # Memory efficiency
        try:
            del self.PCA
        except:
            pass

    # Autoencoder
        if self.datasetToUse in self.autoencoderStage:
            if self.autoencoderInteraction:
                for dataset in self.compilation_data:
                    for feature in columns:
                        for meta in self.autoencoderFeature[dataset].columns:
                            self.compilation_data[dataset]['{}_{}'.format(feature, meta)] = self.compilation_data[dataset][feature].values * self.autoencoderFeature[dataset][meta].values
            else:
                for dataset in self.compilation_data:
                    self.compilation_data[dataset] = pd.concat([self.compilation_data[dataset].reset_index(drop=True), 
                                                                self.autoencoderFeature[dataset].reset_index(drop=True)], axis=1)
        else:
            pass

    # Memory efficiency
        try:
            del self.autoencoderFeature
        except:
            pass

    # Knn distances
        if self.datasetToUse in self.knnStage:
            if self.knnInteraction:
                for dataset in self.compilation_data:
                    for feature in columns:
                        for meta in self.knnDistances[dataset].columns:
                            self.compilation_data[dataset]['{}_{}'.format(feature, meta)] = self.compilation_data[dataset][feature].values * self.knnDistances[dataset][meta].values
            else:
                for dataset in self.compilation_data:
                    self.compilation_data[dataset] = pd.concat([self.compilation_data[dataset].reset_index(drop=True), 
                                                                self.knnDistances[dataset].reset_index(drop=True)], axis=1)
        else:
            pass

    # Memory efficiency
        try:
            del self.knnDistances
        except:
            pass


#----------------------------------------------------------------------------------------------
#---------------------------------------- COMPILATION -----------------------------------------



    def compile(self, nCores, neuralNetworkCompiler=False, hidden=None, dropout=0.5, learningRate=0.0001, 
                batch=64, epoch=2, cvNumber=1, displayStep=10000, useGPU=False):

    # Defining score

        score = make_scorer(score_func = log_loss)
        time = datetime.now()
        
    # Data
        self.finalPrediction = {}
        for dataset in self.compilation_data:
            self.finalPrediction[dataset] = pd.DataFrame()

        if neuralNetworkCompiler:
            
            print('\n---------------------------------------------')
            print('>> Processing compilation [Neural Network]\n')

        # Model
            class NN(nn.Module):

                def __init__(self):
                    super(NN, self).__init__()

                    self.linear = nn.Sequential(
                        nn.Linear(self.compilation_data[self.datasetToUse].shape[1], hidden),
                        nn.Tanh(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden, 1))

                def forward(self, x):
                    out = self.linear(x)
                    return out

        # data

            if self.notYetNN_comp:
                for dataset in self.compilation_data:
                    self.compilation_data[dataset] = np.array(self.compilation_data[dataset])
                    if dataset != 'real_data':
                        self.Ytrain[dataset] = np.array(self.Ytrain[dataset].values)
                        self.Ytrain[dataset] = self.Ytrain[dataset].reshape(self.Ytrain[dataset].shape[0], 1)

                self.compilation_data[self.datasetToUse] = np.concatenate((self.compilation_data[self.datasetToUse], self.compilation_data['test']), axis=0)
                self.Ytrain[self.datasetToUse] = np.concatenate((self.Ytrain[self.datasetToUse], self.Ytrain['test']), axis=0)

                del self.compilation_data['test']
                self.notYetNN_comp = False

        # Tuning hyperparameter

            if self.evaluate:

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
                    net = NN()

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

                print("\nFinal valid log loss: %.5f\n" % cvScore)
                print('\nTotal running time {}'.format(diff(datetime.now(), time)))

        # Final model

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
                net = NN()

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

                print('\nRunning time {}\n'.format(diff(datetime.now(), time)))

    # Machine learning model

        else:

            for name, model, parameters, baggingSteps, nFeatures, stage in zip(self.modelNames, self.models, self.parameters, self.baggingSteps, self.nFeatures, self.stage):
                
                if stage == self.datasetToUse:                                                                                     ## There is normally only one model that fits here !!!!

                    print('\n---------------------------------------------')
                    print('>> Processing compilation [{}]\n'.format(name))

                # Tuning

                    gscv = GridSearchCV(model, parameters, scoring=score, n_jobs=nCores, cv=5)
                    gscv.fit(self.compilation_data[self.datasetToUse], self.Ytrain[self.datasetToUse])                        ## COMPILATION TRAINING ON SECOND STAGE PREDICTION OF XTRAIN3
                                                                                                                              ## IF THERE IS TWO STAGES, ELSE ON XTRAIN2
                # Final prediction

                    for dataset in self.compilation_data:
                        self.finalPrediction[dataset]['final_prediction'] = gscv.predict_proba(self.compilation_data[dataset])[:,1]

            print('\nCompilation running time {}\n'.format(diff(datetime.now(), time)))

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


