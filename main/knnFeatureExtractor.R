

rm(list=ls())
cat('\014')


#---------------------------------------------------------
#-------------------- 0. LIBRARIES -----------------------


# ## Installing
# library(devtools)
# httr::set_config(httr::config( ssl_verifypeer = 0L))  # May be risky
# install_github("davpinto/fastknn")

library(data.table)
library(fastknn)


#---------------------------------------------------------
#-------------------- 1. DATA ----------------------------


week=112
names=c('bernie', 'jordan', 'elizabeth', 'ken', 'charles')
k=25
nCores=32

bigtime = Sys.time()

for (name in names) {
  
  cat('----------------------------------------------------------')
  cat(' Processing', name, '\n')
  
  train = fread(paste0("/home/hugoperrin/Bureau/Datasets/Numerai/w",week,"/numerai_training_data.csv"))
  tournament = fread(paste0("/home/hugoperrin/Bureau/Datasets/Numerai/w",week,"/numerai_tournament_data.csv"))
  
  Ytrain = eval(parse(text=paste0('train[, factor(target_',name,')]')))
  
  train[, `:=` (target_bernie = NULL,
                target_charles = NULL,
                target_elizabeth = NULL,
                target_jordan = NULL,
                target_ken = NULL,
                era = NULL,
                data_type = NULL,
                id = NULL)]
  
  tournament[, `:=` (target_bernie = NULL,
                     target_charles = NULL,
                     target_elizabeth = NULL,
                     target_jordan = NULL,
                     target_ken = NULL,
                     era = NULL,
                     data_type = NULL,
                     id = NULL)]
  
  
  #---------------------------------------------------------
  #-------------------- 2. FEATURE ENGINEERING -------------
  
  
  # Training size
  sampleSize = 50000
  maxSize = 350000
  
  ids = sample(maxSize, sampleSize)
  
  # Training data
  Xtrain = data.matrix(train[ids])
  Ytrain = Ytrain[ids]

  
  # Training prediction
  time = Sys.time()
  
  train = data.matrix(train)
  
  KNNfeatures = knnExtract(xtr = Xtrain, ytr = Ytrain, xte = train, k = k, nthread = nCores)
  
  KNNfeaturesTrain = data.table(KNNfeatures$new.te)[, .(knn5, knn10, knn25, knn30, knn35, knn50)]
  fwrite(KNNfeaturesTrain, paste0("/home/hugoperrin/Bureau/Datasets/Numerai/w",week,"/knnFeatures_train_",name,".csv"), nThread=8)
  
  rm(KNNfeatures, KNNfeaturesTrain, train)
  
  cat("\n>>Training data processing time :", round(difftime(Sys.time(),time,units = c("min")), digits = 2), "mins\n\n")
  
  
  # Tournament prediction
  time = Sys.time()
  
  tournament = data.matrix(tournament)
  
  KNNfeatures = knnExtract(xtr = Xtrain, ytr = Ytrain, xte = tournament, k = k, nthread = nCores)
  
  KNNfeaturesTournament = data.table(KNNfeatures$new.te)[, .(knn5, knn10, knn25, knn30, knn35, knn50)]
  fwrite(KNNfeaturesTournament, paste0("/home/hugoperrin/Bureau/Datasets/Numerai/w",week,"/knnFeatures_tournament_",name,".csv"), nThread=8)
  
  rm(KNNfeatures, KNNfeaturesTournament, tournament)
  
  cat("\n>>Tournament data processing time :", round(difftime(Sys.time(),time,units = c("min")), digits = 2), "mins\n\n")
  
  rm(Xtrain, Ytrain, ids)
}

cat("Total processing time :", round(difftime(Sys.time(),bigtime,units = c("min")), digits = 2), "mins\n\n")


