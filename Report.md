Practical Machine Learning Course Project Report

Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal was to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).
The training data for this project https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv and the test data are available at https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv.

Report Submission
The goal of the project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. This report describes how I built the model, using cross validation, sample error, and the choices. For the project I also used the prediction model to predict 20 different test cases.
Loading the required packages…
> library(caret)
> library(rpart)
> library(rpart.plot)
> library(randomForest)

Setting seed for so results can be reproduced…
> set.seed(12345)
Loading training data…
> trainingFile = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
> trainData = read.csv(trainingFile)
Loading testing data…
> testingFile = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
> testingData = read.csv(testingFile)
The training data contains 19,622robservations and 160 variables. The classe variable in is the outcome.
> dim(trainData)
[1] 19622   160
The testing data contains 20 observations and 160 variables. 
> dim(testingData)
[1]  20 160
Pre-processing training and test data to address near zero values
> NearZeroValues = nearZeroVar(trainData, saveMetrics=TRUE)
> training = trainData[ , !NearZeroValues$nzv]
> testing = testingData[ , !NearZeroValues$nzv]

Pattern-matching using regular expression to remove columns that add little value and contain NA
> regEx = grepl("^X|timestamp|user_name", names(training))
> trainingSet = training[ , !regEx]
> testingSet = testing[ , !regEx]
> condition = (colSums(is.na(trainingSet)) == 0)
> trainingSet = trainingSet[ , condition]
> testingSet = testingSet[ , condition]
> dim(trainingSet)
[1] 19622    54
> dim(testingSet)
[1] 20 54

Partitioning the training set into training and validation, based on 70 / 30 percent…

> inTrain = createDataPartition(trainingSet$classe, p=.7, list=FALSE)
> training = trainingSet[inTrain, ]
> validation = trainingSet[-inTrain, ]
> dim(training)
[1] 13737    54
> dim(validation)
[1] 5885   54
> dim(testingSet)
[1] 20 54

Build a decision tree model for activity prediction and assess accuracy…

> decisionTree = rpart(classe ~ ., data=training, method = "class")
> predictTree = predict(decisionTree, validation, type="class")
> accuracy = postResample(predictTree, validation$classe)
> OSE = 1 - as.numeric(confusionMatrix(validation$classe, predictTree)$overall[1])
> accuracy
 Accuracy     Kappa 
0.7367884 0.6656280 
> OSE
[1] 0.2632116

Build a random forest model for and assess accuracy…

> randomForestModel = train(classe ~ ., data=training, method = "rf", trControl=trainControl(method="cv", 5), ntree=250)
> randomForestPredict = predict(randomForestModel, validation)
> accuracy <- postResample(randomForestPredict, validation$classe)
> OSE = 1 - as.numeric(confusionMatrix(validation$classe, randomForestPredict)$overall[1])
> accuracy
 Accuracy     Kappa 
0.9966015 0.9957009 
> OSE
[1] 0.003398471

Random Forest results outperform decision tree as expected.


Quiz questions

Apply your machine learning algorithm to the 20 test cases available in the test data above and submit your predictions in appropriate format to the Course Project Prediction Quiz for automated grading.

> predict(randomForestModel, testingSet[1, ])
[1] B
Levels: A B C D E
> predict(randomForestModel, testingSet[2, ])
[1] A
Levels: A B C D E
> predict(randomForestModel, testingSet[3, ])
[1] B
Levels: A B C D E
> predict(randomForestModel, testingSet[4, ])
[1] A
Levels: A B C D E
> predict(randomForestModel, testingSet[5, ])
[1] A
Levels: A B C D E
> predict(randomForestModel, testingSet[6, ])
[1] E
Levels: A B C D E
> predict(randomForestModel, testingSet[7, ])
[1] D
Levels: A B C D E
> predict(randomForestModel, testingSet[8, ])
[1] B
Levels: A B C D E
> predict(randomForestModel, testingSet[9, ])
[1] A
Levels: A B C D E
> predict(randomForestModel, testingSet[10, ])
[1] A
Levels: A B C D E
> predict(randomForestModel, testingSet[11, ])
[1] B
Levels: A B C D E
> predict(randomForestModel, testingSet[12, ])
[1] C
Levels: A B C D E
> predict(randomForestModel, testingSet[13, ])
[1] B
Levels: A B C D E
> predict(randomForestModel, testingSet[14, ])
[1] A
Levels: A B C D E
> predict(randomForestModel, testingSet[15, ])
[1] E
Levels: A B C D E
> predict(randomForestModel, testingSet[16, ])
[1] E
Levels: A B C D E
> predict(randomForestModel, testingSet[17, ])
[1] A
Levels: A B C D E
> predict(randomForestModel, testingSet[18, ])
[1] B
Levels: A B C D E
> predict(randomForestModel, testingSet[19, ])
[1] B
Levels: A B C D E
> predict(randomForestModel, testingSet[20, ])
[1] B
Levels: A B C D E
>





