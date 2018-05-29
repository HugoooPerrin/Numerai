rm(list = ls())
cat('\14')

#-----------------------------------------------------------------------
#---------------------------- 1. Librairies ----------------------------

# Data wrangling
library(data.table)
library(readr)

# Machine learning
library(xgboost)
library(mlr)
library(randomForest)
library(ranger)
library(keras)
library(kerasR)
library(pls)
library(glmnet)

# library(tfestimators)  -> promissing but not released on CRAN yet

# Scoring metrics
library(Metrics)

# Parallelised computation
library(parallel)
library(parallelMap)

library(extraTrees) # Need rJava which isn't working

#-----------------------------------------------------------------------
#------------------------------- 2. Data -------------------------------

# Loading
Xtrain = fread("C:/Users/hperrin/Desktop/Numerai/w82/numerai_datasets/numerai_training_data.csv")
Xtest = fread("C:/Users/hperrin/Desktop/Numerai/w82/numerai_datasets/numerai_tournament_data.csv")

# Preprocessing
Xfinal = Xtest
Xtest = Xtest[data_type == 'validation']

Xtest[, `:=` (id = NULL,
              era = NULL,
              data_type = NULL)]

Xtrain[, `:=` (id = NULL,
               era = NULL,
               data_type = NULL)]

final_id = Xfinal[, .(id)]

Xfinal[, `:=` (id = NULL,
               era = NULL,
               data_type = NULL,
               target = NULL)]

#-------------------------------------------------------------------------
#--------------------------- 3. Preprocessing ----------------------------

# # OUTLIERS
# outlier_rows = vector()
# 
# for (feature in setdiff(colnames(Xtrain), 'target'))
# {
#   text = paste0('outlier_rows = unique(c(Xtrain[, which(',feature,' %in% boxplot.stats(',feature,')$out)], outlier_rows))')
#   eval(parse(text = text))
# }


# META FEATURES
metafeature_train = data.table(index = 1:nrow(Xtrain))
metafeature_test = data.table(index = 1:nrow(Xtest))
metafeature_final = data.table(index = 1:nrow(Xfinal))

indiv_mean = Xtrain[,1:50][, apply(.SD, FUN = mean, MARGIN = 2)]

metafeature_train[, distance := Xtrain[,1:50][, apply(.SD, FUN = function(x) {sqrt((x-indiv_mean)%*%(x-indiv_mean))}, MARGIN = 1)]]
metafeature_train[, variance := Xtrain[,1:50][, apply(.SD, FUN = var, MARGIN = 1)]]
metafeature_train[, mean := Xtrain[,1:50][, apply(.SD, FUN = mean, MARGIN = 1)]]

metafeature_test[, distance := Xtest[,1:50][, apply(.SD, FUN = function(x) {sqrt((x-indiv_mean)%*%(x-indiv_mean))}, MARGIN = 1)]]
metafeature_test[, variance := Xtest[,1:50][, apply(.SD, FUN = var, MARGIN = 1)]]
metafeature_test[, mean := Xtest[,1:50][, apply(.SD, FUN = mean, MARGIN = 1)]]

metafeature_final[, distance := Xfinal[, apply(.SD, FUN = function(x) {sqrt((x-indiv_mean)%*%(x-indiv_mean))}, MARGIN = 1)]]
metafeature_final[, variance := Xfinal[, apply(.SD, FUN = var, MARGIN = 1)]]
metafeature_final[, mean := Xfinal[, apply(.SD, FUN = mean, MARGIN = 1)]]


save(metafeature_train, metafeature_test, metafeature_final, file = "C:/Users/hperrin/Desktop/Numerai/w82/metafeature.Rda")


#-------------------------------------------------------------------------
#--------------------------- 4. Keras encoder ----------------------------

encoding = 25

# DATA
Ytrain = Xtrain[, target]
Xtrain = matrix(unlist(Xtrain[,1:50]), nrow = nrow(Xtrain), ncol = 50)

Ytest = Xtest[, target]
Xtest = matrix(unlist(Xtest[,1:50]), nrow = nrow(Xtest), ncol = 50)

Xfinal  = matrix(unlist(Xfinal), nrow = nrow(Xfinal), ncol = 50)

# MODEL
NN_model <- keras_model_sequential() %>% 
  
                          layer_dense(units = 2056, activation = 'relu', input_shape = c(50)) %>% 
                          layer_dropout(rate = 0.9) %>%
                          layer_dense(name = 'encoder', units = encoding, activation = 'tanh') %>% 
                          layer_dropout(rate = 0.5) %>%
                          layer_dense(units = 1, activation = 'sigmoid') %>% 
  
                              compile(loss = 'binary_crossentropy',
                                      optimizer = optimizer_rmsprop(lr = 0.0001, rho = 0.9, epsilon = 1e-08, decay = 0),
                                      metrics = c('accuracy'))


history <- NN_model %>% fit(Xtrain, Ytrain, 
                            epochs = 5, 
                            batch_size = 128,
                            validation_split = 0.2,
                            callbacks = c(callback_early_stopping(monitor = 'val_loss', patience = 2)),
                            verbose = 0)


encoder_model <- keras_model(inputs = NN_model$input,
                             outputs = get_layer(NN_model, 'encoder')$output)

plot(history)

NN_model %>% evaluate(Xtest, Ytest, verbose = 0)

# TRAIN ENCODING
encoded_train = data.table(encoder_model %>% predict(Xtrain, verbose = 0))
encoded_train[, target := Ytrain]

# TEST ENCODING
encoded_test = data.table(encoder_model %>% predict(Xtest, verbose = 0))
encoded_test[, target := Ytest]

# FINAL ENCODING
encoded_final = data.table(encoder_model %>% predict(Xfinal, verbose = 0))

rm(Xtrain, Xtest, Xfinal)

save(encoded_train, encoded_test, encoded_final, encoding, file = "C:/Users/hperrin/Desktop/Numerai/w82/encoded1.Rda")

# write.csv(encoded_train, file = "C:/Users/hperrin/Desktop/Numerai/w82/encoded_train.csv")
# write.csv(encoded_test, file = "C:/Users/hperrin/Desktop/Numerai/w82/encoded_test.csv")
# write.csv(encoded_final, file = "C:/Users/hperrin/Desktop/Numerai/w82/encoded_real_data.csv")


#-------------------------------------------------------------------------
#--------------------------- 5. RandomForest -----------------------------

# load("C:/Users/hperrin/Desktop/Numerai/w81/encoded4.Rda")
n_cores = detectCores()

stacking_train = data.table(index = 1:nrow(encoded_train))
stacking_test = data.table(index = 1:nrow(encoded_test))
stacking_final = data.table(index = 1:nrow(encoded_final))

steps = 15
n_feature = 50

for (i in 1:steps)
{
  time1 = Sys.time()

  # SELECT RANDOMLY SOME VARIABLES
  choosen_var = sample(1:encoding,n_feature, replace = TRUE)
  Xtrain_temp = as.data.frame(encoded_train)[, c(choosen_var,encoding+1)]


  # RUN MODEL
  ranger_model = ranger(target ~ .,
                        data = Xtrain_temp,
                        num.trees = 75,
                        mtry = 10,
                        min.node.size = 50,
                        sample.fraction = 0.7,
                        num.threads = n_cores)

  # TRAIN PREDICTIONS
  eval(parse(text = paste0('stacking_train[, ranger_prob',i,' := predict(ranger_model, Xtrain_temp)$predictions]')))

  # TEST PREDICTIONS
  Xtest_temp = as.data.frame(encoded_test)[, c(choosen_var)]
  eval(parse(text = paste0('stacking_test[, ranger_prob',i,' := predict(ranger_model, Xtest_temp)$predictions]')))

  # FINAL PREDICTIONS
  Xfinal_temp = as.data.frame(encoded_final)[, c(choosen_var)]
  eval(parse(text = paste0('stacking_final[, ranger_prob',i,' := predict(ranger_model, Xfinal_temp)$predictions]')))


  cat('\n\n\nRandomForest step',i,'time : ', difftime(Sys.time(),time1,units = c("min")), "minutes !\n\n\n")

}


stacking_train[, index := NULL]
stacking_test[, index := NULL]
stacking_final[, index := NULL]


save(stacking_train, stacking_test, stacking_final, file = "C:/Users/hperrin/Desktop/Numerai/w81/rf_stacking2.Rda")


#-------------------------------------------------------------------------
#--------------------------- 5. Xgboost ----------------------------------

load("../w82/encoded1.Rda")
encoding = 25
n_cores = detectCores()

stacking_train = data.table(index = 1:nrow(encoded_train))
stacking_test = data.table(index = 1:nrow(encoded_test))
stacking_final = data.table(index = 1:nrow(encoded_final))

Ytrain = encoded_train[, target]

steps = 3
n_feature = 10

fixed_params = list(objective = "binary:logistic", 
                    eval_metric = "logloss", 
                    nrounds = 80, 
                    eta = 0.6, 
                    booster = 'gbtree',
                    max_depth = 4,
                    verbose = 0)

for (i in 1:steps)
{
  time1 = Sys.time()
  
  ##---------------------------------------------------------
  # SELECT RANDOMLY SOME VARIABLES
  choosen_var = sample(1:encoding,n_feature, replace = TRUE)
  Xtrain_temp = as.data.frame(encoded_train)[, c(choosen_var,encoding+1)]  
  
  # PROCESS DATA
  data = xgb.DMatrix(data = as.matrix(Xtrain_temp[, -c(n_feature+1)]),   
                     label = Ytrain)
  
  
  ##--------------------------------------------------------
  # CREATE TASKS
  traintask <- makeClassifTask(data = Xtrain_temp, target = "target")
  
  # CREATE LEARNER
  Learner <- makeLearner("classif.xgboost", predict.type = "prob")
  Learner$par.vals <- fixed_params
  
  # SET PARAMETER SPACE
  parameters <- makeParamSet(makeNumericParam("min_child_weight", lower = 10, upper = 1000), 
                             makeNumericParam("subsample", lower = 0.5, upper = 1),            # Sampling 
                             makeNumericParam("colsample_bytree", lower = 0.1, upper = 1),     # Bagging
                             makeNumericParam("lambda", lower = 0, upper = 1),               # L2 regularization
                             makeNumericParam("alpha", lower = 0, upper = 1))                # L1 regularization
  
  # SET RESAMPLING STRATEGY
  sampling <- makeResampleDesc("CV", stratify = TRUE, iters = 3)
  
  # SET SEARCH STRATEGY
  control <- makeTuneControlRandom(maxit = 16)
  
  parallelStartSocket(cpus = n_cores)
  
  # TUNING
  mytune <- tuneParams(learner = Learner, 
                       task = traintask, 
                       resampling = sampling, 
                       measures = logloss, 
                       par.set = parameters, 
                       control = control, 
                       show.info = FALSE)
  
  ##--------------------------------------------------------
  # GET BEST PARAMETERS FOR THIS STEP
  best_moving_params = mytune$x
  parameters = append(fixed_params, best_moving_params)
  
  # TRAINING MODEL
  model = xgboost(params = parameters,
                  data = data,
                  nrounds = 75)
  
  # TRAIN PREDICTIONS
  eval(parse(text = paste0('stacking_train[, xgb_prob',i,' := predict(model, as.matrix(Xtrain_temp[, -c(n_feature+1)]))]')))
  
  # TEST PREDICTIONS
  Xtest_temp = as.data.frame(encoded_test)[, c(choosen_var)]
  eval(parse(text = paste0('stacking_test[, xgb_prob',i,' := predict(model, as.matrix(Xtest_temp))]')))
  
  # FINAL PREDICTIONS
  Xfinal_temp = as.data.frame(encoded_final)[, c(choosen_var)]
  eval(parse(text = paste0('stacking_final[, xgb_prob',i,' := predict(model, as.matrix(Xfinal_temp))]')))

  
  cat('\n\n\nXgboost step',i,'time : ', difftime(Sys.time(),time1,units = c("min")), "minutes !\n\n\n")
}

stacking_train[, index := NULL]
stacking_test[, index := NULL]
stacking_final[, index := NULL]

save(stacking_train, stacking_test, stacking_final, file = "../w82/stacking_xgb.Rda")


#-------------------------------------------------------------------------
#--------------------------- 5. Multiple regressions ----------------------

load("C:/Users/hperrin/Desktop/Numerai/w81/encoded3.Rda")
encoding = 300
n_cores = detectCores()

stacking_train = data.table(index = 1:nrow(encoded_train))
stacking_test = data.table(index = 1:nrow(encoded_test))
stacking_final = data.table(index = 1:nrow(encoded_final))

Ytrain = encoded_train[, target]
Ytest = encoded_test[, target]

steps = 30
n_feature = 25

for (i in 1:steps)
{
  time1 = Sys.time()
  
  ##---------------------------------------------------------
  # SELECT RANDOMLY SOME VARIABLES
  choosen_var = sample(1:encoding,n_feature, replace = TRUE)
  Xtrain_temp = as.data.frame(encoded_train)[, c(choosen_var)]  
  
  model = lm(Ytrain ~.,
             data = Xtrain_temp)
  
  predicted = predict(model, as.data.frame(encoded_test)[, c(choosen_var)])
  cat('LogLoss : ', logLoss(Ytest, predicted))
  
  # Train prediction
  eval(parse(text = paste0('stacking_train[, linear_prob',i,' := predict(model, Xtrain_temp)]')))
  
  # Test prediction
  eval(parse(text = paste0('stacking_test[, linear_prob',i,' := predicted]')))
  
  # Final prediction
  Xfinal_temp = as.data.frame(encoded_final)[, c(choosen_var)]
  eval(parse(text = paste0('stacking_final[, linear_prob',i,' := predict(model, Xfinal_temp)]')))
  
  cat('\nLinear step',i,'time : ', difftime(Sys.time(),time1,units = c("min")), "minutes !\n")
}

stacking_train[, index := NULL]
stacking_test[, index := NULL]
stacking_final[, index := NULL]

save(stacking_train, stacking_test, stacking_final, file = "../w81/stacking_lineaire1.Rda")


#-------------------------------------------------------------------------
#--------------------------- 6. Feature weighted linear regression -------

# Reload stacking data
load("../w82/stacking_xgb.Rda")

Ytrain = Xtrain[, target]
Ytest = Xtest[, target]

Xtrain = stacking_train
Xtest = stacking_test
Xfinal = stacking_final

# load("C:/Users/hperrin/Desktop/Numerai/w81/stacking_xgb1.Rda")         # BUG WITH RF
# Xtrain = cbind(Xtrain, stacking_train)
# Xtest = cbind(Xtest, stacking_test)
# Xfinal = cbind(Xfinal, stacking_final)

stacking_train = data.table(read_csv('../w82/first_stage_train.csv'))    
stacking_test = data.table(read_csv('../w82/first_stage_test.csv'))       
stacking_final = data.table(read_csv('../w82/first_stage_final.csv'))       

Xtrain = cbind(Xtrain, stacking_train[1:535713])
Xtest = cbind(Xtest, stacking_test)
Xfinal = cbind(Xfinal, stacking_final)

load('../w81/metafeature.Rda')

##--------------------------------------------------------
### SIMPLE MEAN
predicted = apply(Xtest, FUN = mean, MARGIN = 1)
cat('LogLoss : ', logLoss(Ytest, predicted))

final_pred = apply(Xfinal, FUN = mean, MARGIN = 1)

submission = data.table(id = final_id$id,
                       probability = final_pred)

write.csv(submission, file = "../w81/3rd_submit.csv", row.names = FALSE)


##--------------------------------------------------------
### LINEAR REGRESSION :  

# Without metafeature weight
# Training
linear_model = lm(Ytrain ~ .-1, data = Xtrain)

# Metric
predicted = predict(linear_model, Xtest)
cat('LogLoss : ', logLoss(Ytest, predicted))

final_pred = predict(linear_model, Xfinal)

submission= data.table(id = final_id$id,
                       probability = final_pred)

write.csv(submission, file = "../w82/2nd_submit.csv", row.names = FALSE)

# With metafeature weights
# Preprocessing
for (col in colnames(Xtrain))
{
  for (data in c('train', 'test', 'final'))
  {
    for (meta in c('variance', 'mean', 'distance'))
    {
      eval(parse(text = paste0('X',data,'[,',col,'_',meta,' := ',col,' * metafeature_',data,'[,',meta,']]')))
    }
  }
}

# Training
linear_model = lm(Ytrain ~ .-1, data = Xtrain)

# Metric
predicted = predict(linear_model, Xtest)
cat('LogLoss : ', logLoss(Ytest, predicted))

final_pred = predict(linear_model, Xfinal)

submission= data.table(id = final_id$id,
                       probability = final_pred)

write.csv(submission, file = "../w82/3rd_submit.csv", row.names = FALSE)

# Tester glmnet pour du lasso auto selector

