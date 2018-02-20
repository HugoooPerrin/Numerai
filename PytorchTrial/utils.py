

from torch.autograd import Variable

import numpy as np

from sklearn.metrics import log_loss
from scipy.special import expit


# --------------------------------------------------------------
# --------------------------------------------------------------


def train(num_epoch, model, train_loader, optimizer, criterion, display_step=500, valid_loader=None, use_GPU=True):

    if use_GPU:
        model = model.cuda()

    # Count the number of parameters in the network
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('\n>> Learning: {} parameters\n'.format(params))

    i = 0
    for epoch in range(num_epoch):

        running_loss = 0.0
        for data in train_loader:
            model.train()

            # Get the inputs (batch)
            inputs, labels = data
            # Wrap them in Variable
            if use_GPU is True:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)

            # Loss
            loss = criterion(outputs, labels)

            del inputs, labels  # Useful ?

            # Backward 
            loss.backward()

            # Optimize
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            i += 1
            if i % display_step == display_step-1:    

                if valid_loader is not None:
                    model.eval()

                    inputs = valid_loader.dataset.data_tensor
                    labels = valid_loader.dataset.target_tensor
                    labels = labels

                    # Wrap them in Variable
                    if use_GPU is True:
                        inputs = Variable(inputs.cuda(), requires_grad=False)
                    else:
                        inputs = Variable(inputs, requires_grad=False)

                    outputs = model(inputs)
                    prediction = outputs.data.float()             
                    prediction = expit(prediction.cpu().numpy())
                    target = labels.cpu().numpy()    

                    print('Epoch: %d, step: %5d, training loss: %.4f, validation loss: %.4f' % 
                          (epoch + 1, i + 1, running_loss / display_step, log_loss(target, prediction)))
                    running_loss = 0.0
                else:
                    print('Epoch: %d, step: %5d, training loss: %.4f' % 
                          (epoch + 1, i + 1, running_loss / display_step))
                    running_loss = 0.0


def predict(model, dataset_loader, use_GPU=True):

    model.eval()

    concatenate = False                # Prediction by batch to optimize GPU memory use

    # Get the inputs
    for data in dataset_loader:

        if len(data) == 2:
            inputs, other = data       # Small if statement allowing to have loader 
        else:
            inputs = data              # with or without target (validation or test)

        # Wrap them in Variable
        if use_GPU is True:
            inputs = Variable(inputs.cuda(), requires_grad=False)
        else:
            inputs = Variable(inputs, requires_grad=False)

        outputs = model(inputs)

        del inputs

        if use_GPU:
            prediction = outputs.data.cpu().numpy() # probabilities      
        else:
            prediction = outputs.data.numpy() # probabilities      

        if concatenate:
            full_prediction = np.concatenate((full_prediction, prediction), axis=0)
        else:
            full_prediction = prediction
            concatenate = True

    # Compute sigmoid function
    full_prediction = expit(full_prediction)

    return full_prediction
