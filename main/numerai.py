
"""
Main class designed to quickly evaluate different model architectures over Numerai dataset


Models implemented:

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
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Machine Learning
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
import lightgbm as lightgbm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# # Perso
sys.path.append('../')
from pytorch import models, utils

def diff(t_a, t_b):
    t_diff = relativedelta(t_a, t_b)
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

score = make_scorer(score_func = log_loss)


#=========================================================================================================
#================================ NUMERAI CLASS

class Numerai(object):

    def __init__(self, week, firstStageModels, firstStageParameters, 
                             secondStageModel=None, secondStageParameters=None,
                             metafeatures=None, submit=False):
        pass