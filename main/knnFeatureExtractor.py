





"""
Personal set of functions / class  aiming at quickly extract feature from knn:
    - Target mean of k nearest neighbors.
    - Mean distance to k nearest neighbors for each target instance

So 3 new features for every k !
"""





#=========================================================================================================
#================================ MODULES


import pandas as pd
import dask.dataframe as dd
import numpy as np

from scipy.spatial.distance import euclidean

from copy import deepcopy

# If to slow try to replace numpy/pandas by pytorch to rely on GPU computing.


#=========================================================================================================
#================================ FUNCTION


def compute_features(row, Xtrain, Ytrain, k=[]):

    # Copying data set
    final = deepcopy(Ytrain)
    results = {}

    # Computing distances
    final['distance'] = Xtrain.apply(lambda line: euclidean(line, row), axis = 1)

    # Sorting results
    final.sort_values(by='distance', inplace=True)

    # All neighbors
    for number in k:
        results['knnAll{}'.format(number)] = final.iloc[0:number]['target'].values.mean()

    # Only neighbors 1
    for number in k:
        results['knnOnes{}'.format(number)] = final[final['target'] == 1].iloc[0:number]['distance'].values.mean()

    # Only neighbors 0
    for number in k:
        results['knnZeros{}'.format(number)] = final[final['target'] == 0].iloc[0:number]['distance'].values.mean()

    # Returning a dictionnary
    return results


#=========================================================================================================
#================================ CLASS


class KnnFeatureExtractor(object):


    def __init__(self):
        pass


    def fit(self, Xtrain, Ytrain):
        self.Xtrain = deepcopy(Xtrain)

        self.Ytrain = pd.DataFrame()
        self.Ytrain['target'] = Ytrain


    def extract_features(self, data, k=[]):

        return data.apply(lambda row: pd.Series(compute_features(row, self.Xtrain, self.Ytrain, k=k)), axis = 1)


