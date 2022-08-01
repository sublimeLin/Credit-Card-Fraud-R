library(caret)
library(e1071)
library(rpart)
library(randomForest)
library(ggplot2)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(ROSE)
library(DMwR)
library(ROCR)


#Reading the csv file
data <- read.csv(file.choose(),header = T,sep = ",")
head(data)

#Checking for NA value. 
data[is.na(data),]

#No NA value in data

str(data)
summary(data)

#Changing class into factors

data$Class <- factor(data$Class,levels = c("1","0"))

#Class Proportions

table(data$Class)

#data is highly imbalanced, positive class is only 492 out of 284807

prop.table(table(data$Class))


#data partition

set.seed(1234)

Ind <- sample(2,nrow(data),replace = T,prob = c(0.8,0.2))

train <- data[Ind==1,]
test <- data[Ind==2,]

#checking the proportion in  train and test

dim(train)
dim(test)
prop.table(table(train$Class))
prop.table(table(test$Class))



#NAive Bayes -----------------------

library(e1071)
set.seed(1234)

Naive_Model <- naiveBayes(Class ~ .,data = train)

#Make predictions in test data

Naive_Pred <- predict(Naive_Model,test,type = "class")

#Performance Matrics

confusionMatrix(Naive_Pred,test$Class)



#Data balancing

set.seed(1234)
data_smote <- SMOTE(Class ~ .,data,perc.over = 400,perc.under = 150,k=5)
table(data_smote$Class)
prop.table(table(data_smote$Class))

#Splitting new data

set.seed(1234)

Ind_Bal <- sample(2,nrow(data_smote),replace = T,prob = c(0.8,0.2))

train_Bal <- data_smote[Ind_Bal==1,]
test_Bal <- data_smote[Ind_Bal==2,]


#checking the proportion in  train_Bal and test_Bal

dim(train_Bal)
dim(test_Bal)
prop.table(table(train_Bal$Class))
prop.table(table(test_Bal$Class))



#Decision Tree ---------------------------

set.seed(1234)
Tree_Model <- rpart(Class~., data = train_Bal)
summary(Tree_Model)

#Model performance
Tree_Pred <- predict(Tree_Model,test_Bal,type="class")

#Confusion matrice

confusionMatrix(Tree_Pred,test_Bal$Class)
fancyRpartPlot(Tree_Model)

#Variable importance
(Tree_Model$variable.importance)

#ROC curve, f-measure and accuracy
accuracy.meas(Tree_Pred,test_Bal$Class)
roc.curve(Tree_Pred,test_Bal$Class,plotit = T)



# 10-fold cross validation ---------------------------

set.seed(1234)
t_control <-  trainControl(method = "cv", number = 10, savePredictions = TRUE)
Tree_CV <- train(Class ~., data = data_smote, trControl = t_control,method = "rpart",tuneLength=5)
Tree_CV
#Prediction
Tree_CV_Pred <- predict(Tree_CV,test_Bal)

#Confusion matrices
confusionMatrix(Tree_CV_Pred,test_Bal$Class)

#ROC curve, f-measure and accuracy
accuracy.meas(Tree_CV_Pred,test_Bal$Class)
roc.curve(Tree_Pred,test_Bal$Class,plotit = T)




# random forest -----------------------------

# Base Accuracy = 97.4%
# AUC ROC Curve = 97.5%

set.seed(1234)
RF_Model <- randomForest(Class ~ ., data = train_Bal, ntree = 1000, importance = TRUE)
RF_Pred <- predict(RF_Model, test_Bal, type = "class")

# confusion matrix
confusionMatrix(table(RF_Pred, test_Bal$Class))
# variable importance
varImp(RF_Model)

# from package ROSE we get precision/recall and f-measure
accuracy.meas(RF_Pred, test_Bal$Class)
roc.curve(RF_Pred, test_Bal$Class, plotit = T)





