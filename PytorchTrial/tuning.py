
# coding: utf-8

# Import modules

import pandas as pd
import numpy as np
import time

import torch.utils.data as utils
import torch.optim as optim

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Personal modules
from models import *
from utils import *

use_GPU = torch.cuda.is_available()

time1 = time.time()

# Import data
train_vect = pd.read_csv('../../../Datasets/Numerai/w95/numerai_training_data.csv')
test_comments = pd.read_csv('../../../Datasets/Numerai/w95/numerai_tournament_data.csv')

# Preprocess data for torch
test_comments = test_comments[test_comments['data_type'] == 'validation']

labels = np.array(train_vect['target'].values)
labels = labels.reshape(labels.shape[0], 1)
test_labels = np.array(test_comments['target'].values)
test_labels = test_labels.reshape(test_labels.shape[0], 1)

train_vect.drop(['id', 'era', 'data_type', 'target'], inplace=True, axis=1)
test_comments.drop(['id', 'era', 'data_type', 'target'], inplace=True, axis=1)

train_vect = np.array(train_vect)
test_comments = np.array(test_comments)

# Step for 1D convolution
train_vect = train_vect.reshape(train_vect.shape[0],1,train_vect.shape[1])
test_comments = test_comments.reshape(test_comments.shape[0],1,test_comments.shape[1])

time1 = time.time()

# Cross validation loop
CV = 1

CV_score = 0

for i in range(CV):

    print('\n---------------------------------------------------\nLoop number {}'.format(i+1))

    # Train test split
    train_comments, valid_comments, train_labels, valid_labels = train_test_split(train_vect, labels, test_size=0.3)

    # Parameters
    batch_size = 32
    num_epoch = 6

    # Data to tensor
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_comments), 
                                                   torch.FloatTensor(train_labels))

    valid_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(valid_comments), 
                                                   torch.FloatTensor(valid_labels))

    test_dataset = torch.FloatTensor(test_comments)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False, 
                                               num_workers = 8)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers = 8)
    # Model
    net = Inception()

    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimization algorithm
    optimizer = optim.RMSprop(net.parameters(), lr=0.000012, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.9)

    # Training model
    train(num_epoch, net, train_loader, optimizer, criterion, display_step=2000, valid_loader=valid_loader,
          use_GPU=use_GPU)

    # Predicting
    predictions = predict(net, test_loader, use_GPU=use_GPU)
    
    # Performance measuring
    score = log_loss(test_labels, predictions)

    CV_score += score*(1/CV)

    print("\nModel intermediate score: %.6f" % score)


print("\nModel final score: %.6f\n" % CV_score)


time2 = time.time()
diff_time = (time2 - time1)/60
print("Training time is {} minutes\n".format(round(diff_time, 1)))
