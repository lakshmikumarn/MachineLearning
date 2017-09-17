library(caret)
relUrlT <- "~/coursera/datascience/MachineLearning/Week4/pml-training.csv"
relUrlE <- "~/coursera/datascience/MachineLearning/Week4/pml-testing.csv"

trainD <- read.csv(relUrlT,header = TRUE)

## Validation data to be used for out-of-sample validation (typically called testing)
validationD <- read.csv(relUrlE, header = TRUE)


## Create training and testing data partition from from the "training file"
set.seed(12345)
inTrain <- createDataPartition(y=trainD$classe, p=0.7, list = FALSE)
training <- trainD[inTrain,]
testing <- trainD[-inTrain,]

## Try to prune the highly correlated columns. 
## Apply the same for testing and validation data sets to predict the model built based on training data set

trainNZV <- nearZeroVar(training)
training <- training[,-trainNZV]
testing <- testing[,-trainNZV]
validation <- validationD[,-trainNZV]

# I built the model w/o removing NAs 
# naColumns <- sapply(names(training), function(x) all(is.na(training[,x]) == TRUE))
# training <- training[, naColumns==FALSE]
# testing <- testing[, naColumns==FALSE]
# naColumns <- sapply(names(validation), function(x) all(is.na(validation[,x]) == TRUE))
# validation <- validation[,naColumns == FALSE]

# To tune the sampling, create trainControl object to set K-fold cross-validation and add it part of the model building
ctrl <- trainControl(method="cv", number=3, verboseIter=FALSE)
modelFit_rpart <- train(classe ~ ., method = "rpart", data = na.exclude(training), trControl = ctrl)
modelFit_rf <- train(classe ~ ., method = "rf", data = na.exclude(training), prox=TRUE, trControl = ctrl)
modelFit_gbm <- train(classe ~ ., method = "gbm", data = na.exclude(training), verbose= FALSE , trControl = ctrl)

# Predict using the models built on the testing data subset created out of the training data file
pred_rpart <- predict(modelFit_rpart,newdata = na.exclude(testing))
pred_rf <- predict(modelFit_rf,newdata = na.exclude(testing))
pred_gbm <- predict(modelFit_gbm,newdata = na.exclude(testing))

# Check the accuracy results of these models to select a model to predict validation data

print(mean(modelFit_gbm$results$Accuracy))  # [1] 0.9983897
print(mean(modelFit_rpart$results$Accuracy)) #  [1] 0.6065754
print(mean(modelFit_rf$results$Accuracy)) #  [1] 0.9365079

# using gradient boosting model (having higher accuracy results given the sampling)
pred_oos <- predict(modelFit_gbm,newdata = na.exclude(validation[, -ncol(validation)]))

print(pred_oos)

