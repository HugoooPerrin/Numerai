



"""
Running algorithm to search best fitting model


Next steps:
    - Try meta features
    - Neural networks
    - Consistency check
    - Feature interaction / polynomial
    - Using era ?
    - hardcore EDA
"""



#=========================================================================================================
#================================ 0. MODULE


# Numerai class
from numerai import Numerai

# Machine Learning models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import AdaBoostClassifier     # To try
from sklearn.neighbors import KNeighborsClassifier  # To try (Too big dataset ?)


#=========================================================================================================
#================================ 1. CLASS


stacking = Numerai(week=109)


#=========================================================================================================
#================================ 2. FEATURE ENGINEERING


# metafeature = ['variance', 'mean', 'distance']
# stages = [2, 0, 2]
# stacking.add_metafeature(metafeature, stages)


#=========================================================================================================
#================================ 3. MODEL ARCHITECTURE


nCores = -1


models = {'ExtraTrees1':[1, 5, 15, ExtraTreesClassifier(n_jobs = nCores, 
                                                        criterion = 'entropy',
                                                        max_depth = 4,
                                                        bootstrap = True),

                                    {'min_samples_split': [200, 1000],
                                     'n_estimators': [50, 100],                               
                                     'min_samples_leaf': [200, 1000]}],


         'ExtraTrees2': [1, 5, 15, ExtraTreesClassifier(n_jobs = nCores, 
                                                        criterion = 'gini',
                                                        max_depth = 5,
                                                        bootstrap = False),

                                    {'min_samples_split': [200, 1000],
                                     'n_estimators': [50, 100],   
                                     'min_samples_leaf': [200, 1000]}],


         'XGBoost':     [0, 2, 40, XGBClassifier(max_depth = 3, 
                                                 n_estimators = 30,
                                                 nthread = nCores),
 
                                    {'subsample': [0.5, 0.75, 1],                                    
                                     'learning_rate': [0.01, 0.1, 0.5, 1]}],


         'SGDC':        [0, 5, 35, SGDClassifier(loss = 'log', 
                                                 penalty = 'elasticnet', 
                                                 learning_rate = 'optimal',
                                                 max_iter = 10,
                                                 tol = None,
                                                 n_jobs = nCores),

                                    {'alpha': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1],           
                                     'l1_ratio': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]}],


         'LightGBM1':   [1, 15, 15, LGBMClassifier(objective = 'binary',
                                                  max_depth = 3,
                                                  reg_lambda = 0.01,
                                                  n_jobs = nCores), 

                                    {'n_estimators': [25, 50, 100],                                  
                                     'min_child_samples': [50, 500, 1000],
                                     'num_leaves': [128, 1024]}],


         'LightGBM2':   [2, 5, 25, LGBMClassifier(objective = 'binary',
                                                  max_depth = 3,
                                                  reg_lambda = 0.001,
                                                  n_jobs = nCores), 

                                    {'n_estimators': [25, 50, 100],                                  
                                     'min_child_samples': [50, 500, 1000],
                                     'num_leaves': [128, 512, 1024]}]}


stacking.add_model(models)


#=========================================================================================================
#================================ 4. TRAINING MODEL


stacking.fit_tune(nCores, evaluate=True, interaction=None)


#=========================================================================================================
#================================ 5. PREDICTION


# stacking.submit(nCores=-1, submissionNumber=2, week=109)


