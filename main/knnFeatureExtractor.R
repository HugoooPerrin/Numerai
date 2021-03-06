

rm(list=ls())
cat('\014')


#---------------------------------------------------------
#-------------------- 0. MODULES -------------------------


# ## Installing
# library(devtools)
# httr::set_config(httr::config(ssl_verifypeer = 0L))
# install_github("davpinto/fastknn")

library(data.table)
library(fastknn)

"https://github.com/davpinto/fastknn"
"https://sites.google.com/site/aslugsguidetopython/data-analysis/pandas/calling-r-from-python"


#---------------------------------------------------------
#-------------------- 1. DATA ----------------------------


week=113
names=c('bernie', 'ken', 'charles') #, 'jordan', 'elizabeth', 'ken', 'charles')
k=100
nCores=8

bigtime = Sys.time()

for (name in names) {
  
  cat('----------------------------------------------------------')
  cat(' Processing', name, '\n\n')
  
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
  
  
  ## Training parameters
  sampleSize = 20000
  maxSize = 350000
  
  ids = sample(maxSize, sampleSize)
  # Indexing starts at 0 in python while 1 in R
  write.csv(ids-1, paste0("/home/hugoperrin/Bureau/Datasets/Numerai/w",week,"/train_ids_",name,".csv"), row.names=FALSE) 

  ## Training data
  Xtrain = data.matrix(train[ids])
  Ytrain = Ytrain[ids]
  
  
  ## Training prediction
  time = Sys.time()
  
  train = data.matrix(train)
  
  KNNfeatures = knnExtract(xtr = Xtrain, ytr = Ytrain, xte = train, k = k, nthread = nCores)
  
  KNNfeaturesTrain = data.table(KNNfeatures$new.te)[, .(knn10, knn25, knn50, knn100, knn110, knn125, knn150, knn200)]
  
  fwrite(KNNfeaturesTrain, 
    paste0("/home/hugoperrin/Bureau/Datasets/Numerai/w",week,"/knnFeatures_train_",name,".csv"), 
    nThread=8)
  
  rm(KNNfeatures, KNNfeaturesTrain, train)
  
  cat("\n>>Training data processing time :", round(difftime(Sys.time(),time,units = c("min")), digits = 2), "mins\n\n")
  
  
  ## Tournament prediction
  time = Sys.time()
  
  tournament = data.matrix(tournament)
  
  KNNfeatures = knnExtract(xtr = Xtrain, ytr = Ytrain, xte = tournament, k = k, nthread = nCores)
  
  KNNfeaturesTournament = data.table(KNNfeatures$new.te)[, .(knn10, knn25, knn50, knn100, knn110, knn125, knn150, knn200)]
  
  fwrite(KNNfeaturesTournament,
    paste0("/home/hugoperrin/Bureau/Datasets/Numerai/w",week,"/knnFeatures_tournament_",name,".csv"),
    nThread=8)
  
  rm(KNNfeatures, KNNfeaturesTournament, tournament)
  
  cat("\n>>Tournament data processing time :", round(difftime(Sys.time(),time,units = c("min")), digits = 2), "mins\n\n")
  
  rm(Xtrain, Ytrain, ids)
}

cat("Total processing time :", round(difftime(Sys.time(),bigtime,units = c("min")), digits = 2), "mins\n\n")


