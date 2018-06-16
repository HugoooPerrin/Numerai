


#=========================================================================================================
#================================ MODULES


from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier


#=========================================================================================================
#================================ MODEL ARCHITECTURE



models = {'ExtraTrees1':[1, 5, 10, ExtraTreesClassifier(n_jobs = 1, 
                                                         criterion = 'entropy',
                                                         max_depth = 3,
                                                         bootstrap = True),

                                    {'min_samples_split': [250, 1000],
                                     'n_estimators': [25, 50],                               
                                     'min_samples_leaf': [250, 1000]}],


         'ExtraTrees2': [1, 5, 10, ExtraTreesClassifier(n_jobs = 1, 
                                                         criterion = 'gini',
                                                         max_depth = 3,
                                                         bootstrap = True),

                                    {'min_samples_split': [250, 1000],
                                     'n_estimators': [25, 50],                               
                                     'min_samples_leaf': [250, 1000]}],


         'XGBoost':     [0, 1, 50, XGBClassifier(max_depth = 3, 
                                                 n_estimators = 30,
                                                 nthread = 1),
 
                                    {'subsample': [0.5, 0.75, 1],                                    
                                     'learning_rate': [0.01, 0.1, 0.5, 1]}],


         'SGDC':        [0, 1, 20, SGDClassifier(loss = 'log',
                                                 penalty = 'elasticnet',
                                                 learning_rate = 'optimal',
                                                 max_iter = 5,
                                                 tol = None,
                                                 n_jobs = 1),

                                    {'alpha': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1],
                                     'l1_ratio': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]}],


         'LightGBM1':   [1, 5, 10, LGBMClassifier(objective = 'binary',
                                                   max_depth = 3,
                                                   verbose=-1,
                                                   n_jobs = 1), 

                                    {'n_estimators': [25, 50],
                                     'min_child_samples': [100, 500],
                                     'reg_lambda': [0.01, 0.1],
                                     'num_leaves': [8, 16, 32]}],


         'LightGBM2':   [0, 2, 40, LGBMClassifier(objective = 'binary',
                                                  max_depth = 4,
                                                  verbose=-1,
                                                  n_jobs = 1), 

                                   {'n_estimators': [25, 50],                                  
                                     'min_child_samples': [100, 500],
                                     'reg_lambda': [0.01, 0.1],
                                     'num_leaves': [8, 16, 32]}],


         'LightGBM3':   [2, 1, 5, LGBMClassifier(objective = 'binary',
                                                  max_depth = 3,
                                                  verbose=-1,
                                                  n_jobs = 1),

                                    {'n_estimators': [25, 50],                                  
                                     'min_child_samples': [100, 500],
                                     'reg_lambda': [0.01, 0.1],
                                     'num_leaves': [8, 16, 32]}],

         'Catboost':    [0, 2, 25, CatBoostClassifier(loss_function='Logloss',
                                                       verbose=False),

                                    {'iterations': [2, 5],
                                     'learning_rate': [0.1, 1],
                                     'depth': [2, 3, 5],
                                     'l2_leaf_reg': [0.001, 0.01, 0.1]}]}