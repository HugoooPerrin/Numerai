





"""
Main class designed to quickly evaluate different model architectures over Numerai dataset

"""




#=========================================================================================================
#================================ MODULE


# General
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Preprocessing
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Perso
# sys.path.append('../')
# from pytorch import models, utils

# Utils
def diff(t_a, t_b):
    t_diff = relativedelta(t_a, t_b)
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


#=========================================================================================================
#================================ NUMERAI CLASS


class Numerai(object):



    def __init__(self, stageNumber=1):

        self.stageNumber = stageNumber

        self.modelNames = []
        self.models = []
        self.parameters = []
        self.baggingSteps = []
        self.nFeatures = []
        self.stage = []



    def load_data(self, week):
        print('\n---------------------------------------------')
        print('>> Loading data', end='...')
        # Xtrain = pd.read_csv("../../../Datasets/Numerai/w{}/numerai_training_data.csv".format(week))
        # Xvalid = pd.read_csv("../../../Datasets/Numerai/w{}/numerai_tournament_data.csv".format(week))

        Xtrain = pd.read_csv("../../Data/numerai_training_data.csv")
        Xvalid = pd.read_csv("../../Data/numerai_tournament_data.csv")

        real_data = Xvalid.copy(True)
        self.ids = Xvalid['id']

        Xvalid = Xvalid[Xvalid['data_type'] == 'validation']

        Ytrain = Xtrain['target']
        Yvalid = Xvalid['target']

        Xtrain.drop(['id', 'era', 'data_type', 'target'], inplace = True, axis = 1)
        Xvalid.drop(['id', 'era', 'data_type', 'target'], inplace = True, axis = 1)
        real_data.drop(['id', 'era', 'data_type', 'target'], inplace = True, axis = 1)

        if self.stageNumber == 1:

            Xtrain1, Xtrain2, Ytrain1, Ytrain2 = train_test_split(Xtrain, Ytrain, test_size=0.4)
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
                           'valid': Xvalid,
                           'real_data': real_data}

            self.Ytrain = {1: Ytrain1,
                           2: Ytrain2,
                           'valid': Yvalid}

        elif self.stageNumber == 2:

            Xtrain1, Xtrain2, Ytrain1, Ytrain2 = train_test_split(Xtrain, Ytrain, test_size=0.6)
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
                           'valid': Xvalid,
                           'real_data': real_data}

            self.Ytrain = {1: Ytrain1,
                           2: Ytrain2,
                           3: Ytrain3,
                           'valid': Yvalid}



    def add_metafeature(self, name, stage):

        self.metafeature = {}
        for dataset in self.Xtrain:
            self.metafeature[dataset] = pd.DataFrame()

        self.metafeatureName = name
        self.metafeatureStage = stage

    ## Simple metafeature

        if 'variance' in name:
            for dataset in self.Xtrain:
                self.metafeature[dataset]['variance'] = self.Xtrain[dataset].std(axis = 1)

        if 'mean' in name:
            for dataset in self.Xtrain:
                self.metafeature[dataset]['mean'] = self.Xtrain[dataset].mean(axis = 1)

        if 'distance' in name:
            mean_indiv = pd.concat(self.Xtrain.values(), axis=0).mean(axis = 0)
            for dataset in self.Xtrain:
                self.metafeature[dataset]['distance'] = self.Xtrain[dataset].apply(lambda row: euclidean(row, mean_indiv), axis = 1)

    ## Neural Network autoencoder

        if 'autoencoder' in name:
            pass

    ## Mean encoding

        if 'meanEncoding' in name:
            pass

    ## PCA

        if 'PCA' in name:
            pass



    def add_model(self, name, model, parameters, baggingSteps, nFeatures, stage): 
        self.modelNames.append(name)
        self.models.append(model)
        self.parameters.append(parameters)
        self.baggingSteps.append(baggingSteps)
        self.nFeatures.append(nFeatures)
        self.stage.append(stage)



    def add_neuralnetwork(self, stage):
        pass



    def fit_tune(self, nCores=-1):

        score = make_scorer(score_func = log_loss)
        time = datetime.now()
        
    # First stage

        print('\n---------------------------------------------')
        print('>> Processing first stage\n')

        features = [name for name in self.Xtrain[1].columns]
        time1 = datetime.now()

        self.firstStagePrediction = {}
        for dataset in self.Xtrain:
            self.firstStagePrediction[dataset] = pd.DataFrame()

        for name, model, parameters, baggingSteps, nFeatures, stage in zip(self.modelNames, self.models, self.parameters, self.baggingSteps, self.nFeatures, self.stage):
            
            if stage == 1:
                print('>> Processing {}\n'.format(name))

                for step in range(baggingSteps):
        
                    time2 = datetime.now()
                    print("Step {}".format(step+1), end = '...')

                # Creating data
                    np.random.shuffle(features)

                    inter = {}
                    for dataset in self.Xtrain:
                        inter[dataset] = self.Xtrain[dataset][features[:nFeatures]]

                # Adding metafeatures
                    try:
                        for feature, stage in zip(self.metafeatureName, self.metafeatureStage):
                            if stage == 1:
                                for dataset in self.Xtrain:
                                    inter[dataset][feature] = self.metafeature[dataset][feature]
                    except:
                        pass

                # Tuning
                    gscv = GridSearchCV(model, parameters, scoring = score, n_jobs = nCores)
                    gscv.fit(inter[1], self.Ytrain[1])                                                            ## FIRST STAGE TRAINING ON XTRAIN1

                # Saving best predictions
                    for dataset in self.Xtrain:
                        self.firstStagePrediction[dataset]['{}_prediction_{}'.format(name, step+1)] = gscv.predict_proba(inter[dataset])[:,1]

                    print('done in {}'.format(diff(datetime.now(), time2)))
                    print('log loss : {}\n'.format(log_loss(self.Ytrain['valid'], self.firstStagePrediction['valid']['{}_prediction_{}'.format(name,step+1)])))

        print('\nFirst stage running time {}'.format(diff(datetime.now(), time1)))

    # Second stage

        print('\n\n---------------------------------------------')
        print('>> Processing second stage\n')

        if self.stageNumber == 2:

            features = [name for name in self.firstStagePrediction[1].columns]
            time1 = datetime.now()

            self.secondStagePrediction = {}
            for dataset in self.Xtrain:
                self.secondStagePrediction[dataset] = pd.DataFrame()

            for name, model, parameters, baggingSteps, nFeatures, stage in zip(self.modelNames, self.models, self.parameters, self.baggingSteps, self.nFeatures, self.stage):
            
                if stage == 2:
                    print('>> Processing {}\n'.format(name))

                    for step in range(baggingSteps):
            
                        time2 = datetime.now()
                        print("Step {}".format(step+1), end = '...')

                    # Creating data
                        np.random.shuffle(features)

                        inter = {}
                        for dataset in self.Xtrain:
                            inter[dataset] = self.firstStagePrediction[dataset][features[:nFeatures]]

                    # Adding metafeatures
                        try:
                            for feature, stage in zip(self.metafeatureName, self.metafeatureStage):
                                if stage == 2:
                                    for dataset in self.Xtrain:
                                        inter[dataset][feature] = self.metafeature[dataset][feature]
                        except:
                            pass

                    # Tuning
                        gscv = GridSearchCV(model, parameters, scoring = score, n_jobs = nCores)
                        gscv.fit(inter[2], self.Ytrain[2])                                                        ## SECOND STAGE TRAINING ON FIRST STAGE PREDICTION OF XTRAIN2
                                                                                                                  
                    # Saving best predictions
                        for dataset in self.Xtrain:
                            self.secondStagePrediction[dataset]['{}_prediction_{}'.format(name, step+1)] = gscv.predict_proba(inter[dataset])[:,1]

                        print('done in {}'.format(diff(datetime.now(), time2)))
                        print('log loss : {}\n'.format(log_loss(self.Ytrain['valid'], self.secondStagePrediction['valid']['{}_prediction_{}'.format(name,step+1)])))

            print('Second stage running time {}'.format(diff(datetime.now(), time1)))

    # Compilation

        print('\n\n---------------------------------------------')
        print('>> Processing compilation\n')

        self.finalPrediction = {}
        for dataset in self.Xtrain:
            self.finalPrediction[dataset] = pd.DataFrame()

        if self.stageNumber == 1:
            compilation_data = self.firstStagePrediction
            datasetToUse = 2
        elif self.stageNumber == 2:
            compilation_data = self.secondStagePrediction
            datasetToUse = 3

        # Adding metafeatures
        try:
            for feature, stage in zip(self.metafeatureName, self.metafeatureStage):
                if stage == 3:
                    for dataset in self.Xtrain:
                        inter[dataset][feature] = self.metafeature[dataset][feature]
        except:
            pass

        # Tuning
        for name, model, parameters, baggingSteps, nFeatures, stage in zip(self.modelNames, self.models, self.parameters, self.baggingSteps, self.nFeatures, self.stage):
            
            if stage == datasetToUse:                                                                                     ## There is normally only one model that fits here !!!!

                gscv = GridSearchCV(self.compile_model, self.compile_parameters, scoring = score, n_jobs = nCores)
                gscv.fit(compilation_data[datasetToUse], self.Ytrain[datasetToUse])                                       ## COMPILATION TRAINING ON SECOND STAGE PREDICTION OF XTRAIN3
                                                                                                                          ## IF THERE IS TWO STAGES, ELSE ON XTRAIN2
                # Final prediction
                for dataset in self.Xtrain:
                    self.finalPrediction[dataset]['final_prediction'.format(name, step+1)] = gscv.predict_proba(inter[dataset])[:,1]

        print('Total running time {}'.format(diff(datetime.now(), time)))
        print('Final log loss : {}\n'.format(log_loss(self.Ytrain['valid'], self.finalPrediction['valid']['final_prediction'.format(name,step+1)])))                        



    def submit(self, submissionNumber, week):
        submit = pd.DataFrame()
        submit['id'] = self.ids
        submit['probability'] = self.finalPrediction['real_data']

    # Saving prediction
        submit.to_csv('../../../Datasets/Numerai/{0}/submission{1}.csv'.format(week, submissionNumber), index = False)

    # Automated submission through the numerai API
        pass


