library(ggplot2) 
library(readr) 
library(caret)
library(DMwR) 
library(xgboost)
library(Matrix)
library(reshape) 
library(pROC) 

#Loading data set
data <- read.csv(file.choose(), header = T)
head(data)
data$Class <- as.factor(data$Class)

#Checking the proportion of "Fraud detected (1)" & "Fraud not detected (0)"
table(data$Class)
prop.table(table(data$Class))

#Checking for NA value. 
data[is.na(data),]

#No NA's found in data. That's brilliant. Removing the time column 
data$Time <- NULL

#Splitting data into train (60%), test (20%) & CV(20%)
set.seed(1234)
Ind <- createDataPartition(y = data$Class,p=0.75,list =F)
train <- data[Ind,]
test <- data[-Ind,]

test_predict <-(test$Class)

#Changing the Class coulmn to factor
train$Class <- as.factor(train$Class)
table(test$Class)
test$Class <- as.factor(test$Class)

#Using SMOTE func for class imbalancement
i <- grep("Class",colnames(train))
i

data_smote <- SMOTE(Class ~ .,as.data.frame(data),perc.over = 20000,perc.under = 100,k=5, dup_size = 0)
table(data_smote$Class)

Ind_smote <- createDataPartition(y = data_smote$Class,p=0.75,list =F)
train_smote <- data_smote[Ind_smote,]
test_smote <- data_smote[-Ind_smote,]

table(test_smote$Class)

#Prepare for XGBoost

#Back to numeric

train$Class <- as.numeric(levels(train$Class))[train$Class]
train_smote$Class <- as.numeric(levels(train_smote$Class))[train_smote$Class]

test$Class <- as.numeric(levels(test$Class))[test$Class]
test_smote$Class <- as.numeric(levels(test_smote$Class))[test_smote$Class]


#Matrix

train <- Matrix(as.matrix(train),sparse = T)
train
train_smote <- Matrix(as.matrix(train_smote),sparse = T)

test <- Matrix(as.matrix(test),sparse = T)
test_smote <- Matrix(as.matrix(test_smote),sparse = T)

#create XGB Matrices

train_xgb <- xgb.DMatrix(data = train[,-i], label = train[,i])
train_smote_xgb <- xgb.DMatrix(data = train_smote[,-i], label = train_smote[,i])
test_xgb <- xgb.DMatrix(data = test[,-i], label = test[,i])
test_smote_xgb <- xgb.DMatrix(data = test_smote[,-i], label = test_smote[,i])

#Watchlist

watchlist <- list(train = train_xgb)

#Set parameters

parameters <- list(
  # General Parameters
  booster            = "gbtree",          
  silent             = 0,                 
  # Booster Parameters
  eta                = 0.3,               
  gamma              = 0,                 
  max_depth          = 6,                 
  min_child_weight   = 1,                 
  subsample          = 1,                 
  colsample_bytree   = 1,                 
  colsample_bylevel  = 1,                 
  lambda             = 1,                 
  alpha              = 0,                 
  # Task Parameters
  objective          = "binary:logistic",   
  eval_metric        = "auc",
  seed               = 1234               
)


#Model

xgb_model <- xgb.train(parameters, train_xgb,nrounds = 50,watchlist,early_stopping_rounds = 10) 
melted <- melt(xgb_model$evaluation_log,id.vars = "iter")
ggplot(data = melted, aes(x= iter, y= value)) + geom_line() 

xgb_smote_model <- xgb.train(parameters, train_smote_xgb,nrounds = 50,watchlist,early_stopping_rounds = 10) 
melted <- melt(xgb_smote_model$evaluation_log,id.vars = "iter")
ggplot(data = melted, aes(x= iter, y= value)) + geom_line() 

#prediction

q <- 0.5

xgb_predict <- predict(xgb_smote_model,test_smote_xgb)
xgb_predictboolean <- ifelse(xgb_predict >= q,1,0)
roc <- roc(test_smote[,i],predict(xgb_smote_model,test_smote_xgb,type = "prob"))
roc
plot(roc)

xgb_cm <- table(xgb_predictboolean,test_smote$Class)



