
# coding: utf-8

# Import modules

import pandas as pd
import numpy as np
import sys
import nltk
import pickle
import time

import torch
import torch.nn as nn
import torch.utils.data as utils
import torchwordemb
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import log_loss
from multiprocessing import Pool

# Personal modules
from models import *
from utils import *

time1 = time.time()

# Import data
train_vect = pd.read_csv('../../../Datasets/Numerai/w95/numerai_training_data.csv')
test_comments = pd.read_csv('../../../Datasets/Numerai/w95/numerai_tournament_data.csv')

# Preprocess data for torch
test_comments = test_comments[test_comments['data_type'] == 'validation']

labels = train_vect['target']
test_labels = test_comments['target']

train_vect.drop(['id', 'era', 'data_type', 'target'], inplace = True, axis = 1)
test_comments.drop(['id', 'era', 'data_type', 'target'], inplace = True, axis = 1)

# Step for 1D convolution
train_vect = train_vect.reshape(train_vect.shape[0],1,train_vect.shape[1])
test_comments = test_comments.reshape(test_comments.shape[0],1,test_comments.shape[1])

time1 = time.time()

# Cross validation loop
CV = 4

CV_score = 0

for i in range(CV):

    print('\n---------------------------------------------------\nLoop number {}'.format(i+1))

    random_order = permutation(len(train_vect))
    split = floor(len(train_vect)*80/100)

    ## Train test split
    train_comments = train_vect[random_order[:split]]
    valid_comments = train_vect[random_order[split:]]

    train_labels = labels[random_order[:split]]
    valid_labels = labels[random_order[split:]]

    ## Parameters
    use_GPU = True
    batch_size = 64
    num_epoch = 9

    ## Data to tensor
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_comments), 
                                                   torch.FloatTensor(train_labels))

    valid_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(valid_comments), 
                                                   torch.FloatTensor(valid_labels))

    test_dataset = torch.FloatTensor(test_comments)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers = 8)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False, 
                                               num_workers = 8)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False, 
                                               num_workers = 8)
    ## Model 
    net = NN()

    ## Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    ## Optimization algorithm
    optimizer = optim.RMSprop(net.parameters(), lr=0.0003, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.9)

    ## Training model
    train(num_epoch, net, train_loader, optimizer, criterion,
                      display_step=1000, valid_loader=valid_loader, use_GPU=use_GPU)

    ## Predicting 
    predictions = pd.DataFrame(predict_autoencoder(net, test_loader, use_GPU=use_GPU))
    
    ## Performance measuring
    score = mean_squared_error(test_labels, predictions)

    CV_score += score*(1/CV)

    print("\nModel intermediate score: %.4f" % (score))


print("\nModel final score: %.4f\n" % (CV_score))


time2 = time.time()
diff_time = (time2 - time1)/60
print("Training time is {} minutes\n".format(round(diff_time,1)))