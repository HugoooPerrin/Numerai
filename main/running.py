





"""
Running algorithm to search best fitting model
"""




#=========================================================================================================
#================================ 0. MODULE

# Numerai class
from numerai import Numerai

# Machine Learning models
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


#=========================================================================================================
#================================ 1. DATA


stacking = Numerai(stageNumber=1)
stacking.load_data(109)


#=========================================================================================================
#================================ 2. FEATURE ENGINEERING


# metafeature = ['variance', 'mean', 'distance']
# stages = [2, 0, 2]
# stacking.add_metafeature(metafeature, stages)


#=========================================================================================================
#================================ 3. MODEL ARCHITECTURE


nCores = 3


modelNames = ['ExtraTrees1',
              'ExtraTrees2',
              'XGBoost', 
              'SGDC',
              'Lightgbm']

models = [ExtraTreesClassifier(n_jobs = nCores, 
                               criterion = 'entropy',
                               max_depth = 4,
                               n_estimators = 100,
                               bootstrap = True),
          
          ExtraTreesClassifier(n_jobs = nCores, 
                               criterion = 'gini',
                               max_depth = 4,
                               n_estimators = 100,
                               bootstrap = True),
          
#           XGBClassifier(learning_rate = 0.5, 
#                         max_depth = 3, 
#                         n_estimators = 75,
#                         nthread = nCores),
          
          SGDClassifier(loss = 'log', 
                        penalty = 'elasticnet', 
                        learning_rate = 'optimal',
                        n_jobs = nCores),
         
          LGBMClassifier(objective = 'binary',
                         max_depth = 4,
                         n_jobs = nCores)]

parameters = [{'min_samples_split' : [200, 1000],                               # ExtraTreesClassifier entropy
               'min_samples_leaf' : [200, 1000]},

              {'min_samples_split' : [200, 1000],                               # ExtraTreesClassifier gini
               'min_samples_leaf' : [200, 1000]},

#               {'subsample' : [0.5, 0.75, 1]},                                   # XGBoostClassifier

              {'alpha' : [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1],            # SGDClassifier
               'l1_ratio' : [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]},

              {'n_estimators': [50, 100, 200],                                  # Lightgbm
               'num_leaves ' : [15, 40, 100, 500],
               'min_samples_leaf' : [200, 1000],
               'reg_lambda' : [0.001, 0.01, 0.05, 0.1, 0.5, 1]}]

nFeatures = [15, 15, 35, None]

baggingSteps = [5, 5, 5, 1]

stages = [1, 1, 1, 2]


for name, model, parameters, baggingSteps, nFeatures, stage in zip(modelNames, models, parameters, baggingSteps, nFeatures, stages):
    stacking.add_model(name, model, parameters, baggingSteps, nFeatures, stage)


#=========================================================================================================
#================================ 4. TRAINING MODEL


stacking.fit_tune(nCores)


#=========================================================================================================
#================================ 5. PREDICTION


# stacking.submit(submissionNumber=1, week=109)