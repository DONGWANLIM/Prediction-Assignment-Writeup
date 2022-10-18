---
title: "Prediction Assignment Writeup"
author: "DONG WAN LIM"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Overview


### What you should submit

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Peer Review Portion
Your submission for the Peer Review portion should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).

Course Project Prediction Quiz Portion
Apply your machine learning algorithm to the 20 test cases available in the test data above and submit your predictions in appropriate format to the Course Project Prediction Quiz for automated grading.


### Reproducibility

Due to security concerns with the exchange of R code, your code will not be run during the evaluation by your classmates. Please be sure that if they download the repo, they will be able to view the compiled HTML version of your analysis.



## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


### Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.



## Load data and libraries

```{r message=FALSE, warning=FALSE}
library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
library(corrplot)
library(rattle)
```

```{r message=TRUE, warning=FALSE, comment=""}
set.seed(1234)
traincsv <- read.csv("C:/Users/limdw/Downloads/Data Science -  Statistics and Machine Learning Specialization(Johns Hopkins University)/Course 3 - Practical Machine Learning/pml-training.csv")
testcsv <- read.csv("C:/Users/limdw/Downloads/Data Science -  Statistics and Machine Learning Specialization(Johns Hopkins University)/Course 3 - Practical Machine Learning/pml-testing.csv")

dim(traincsv)
dim(testcsv)
```

As we can see, there are 160 variables with 19622 observations in the training set while test set was just 20.



## Clean the data

```{r message=TRUE, warning=FALSE, comment=""}
# removing na columns mostly
traincsv <- traincsv[,colMeans(is.na(traincsv)) < .9]

# removing metadata which is irrelevant to the outcome
traincsv <- traincsv[,-c(1:7)]
```

```{r nzv}
# removing near zero variance variables
nzv <- nearZeroVar(traincsv)
traincsv <- traincsv[,-nzv]
dim(traincsv)
```

```{r}
# check for near zero values in training data
train.nzv <- nzv(traincsv[,-ncol(traincsv)],saveMetrics=TRUE)

# none found so display and count variables submitted for the train function
rownames(train.nzv)
```

```{r}
dim(train.nzv)[1]
```



Now that we have finished removing the unnecessary variables, we can now split the training set into a **validation** and **sub training set**. The testing set “testcsv” will be left alone, and used for the final quiz test cases.

```{r warning=FALSE}
inTrain <- createDataPartition(y = traincsv$classe, p = 0.7, list = F)
train <- traincsv[inTrain,]
valid <- traincsv[-inTrain,]
```



## Create and Test the Models

Here we will test a few popular models including: **Decision Trees, Random Forest, Gradient Boosted Trees, and SVM**. This is probably more than we will need to test, but just out of curiosity and good practice we will run them for comparison.


SEt up the control for training to use 3-fold cross validation.

```{r}
control <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
```



## Decision Tree

```{r message=TRUE, warning=FALSE}
decision_tree <- train(classe~., data = train, method = "rpart",
                       trControl = control, tuneLength = 5)
fancyRpartPlot(decision_tree$finalModel)
```

```{r message=TRUE, warning=FALSE, comment=""}
prediction_tree <- predict(decision_tree, valid)
cm_tree <- confusionMatrix(prediction_tree, factor(valid$classe))
cm_tree
```



## Support Vector Machine

```{r message=TRUE, warning=FALSE, comment="", cache=TRUE}
mod_svm <- train(classe ~ ., data = train, method = "svmLinear", trControl = control, tuneLength = 5, verbose = F)

pred_svm <- predict(mod_svm, valid)
cmsvm <- confusionMatrix(pred_svm, factor(valid$classe))
cmsvm
```



## Gradient Boosted Trees

```{r message=TRUE, warning=FALSE, comment="", cache=TRUE}
mod_gbm <- train(classe ~ ., data = train, method = "gbm", trControl = control, tuneLength = 5, verbose = FALSE)

pred_gbm <- predict(mod_gbm, valid)
cmgbm <- confusionMatrix(pred_gbm, factor(valid$classe))
cmgbm
```



## Random Forest

```{r message=TRUE, warning=FALSE, comment="", cache=TRUE}
mod_rf <- train(classe ~ ., data = train, method = "rf", trControl = control, tuneLength = 5)

pred_rf <- predict(mod_rf, valid)
cmrf <- confusionMatrix(pred_rf, factor(valid$classe))
cmrf
```



## Results

```{r echo=FALSE, message=TRUE, warning=FALSE, comment=""}
models <- c("Tree", "Support Vector Machine", "Gradient Boosted Trees", "Random Forest")
accuracy <- round(c(cm_tree$overall[1], cmsvm$overall[1], cmgbm$overall[1], cmrf$overall[1]),3)
outofsample_error <- 1 - accuracy

data.frame(accuracy = accuracy, outofsample_error = outofsample_error, row.names = models)
```

The best model is the Random Forest model with 0.995 of accuracy and 0.005 of 0.005 out of sample error rate. We found that to be a sufficient enough model to use for our test sets.



## Predictions on test set

The accuracy of the model by predicting with the Validation/Quiz set supplied in the test file.

```{r message=TRUE, warning=FALSE, echo=FALSE, comment=""}
print(predict(mod_rf, testcsv))
```



## Appendix

Correlation matrix of variables in training set

```{r}
corrPlot <- cor(train[, -length(names(train))])
corrplot(corrPlot, method="color")
```

Plotting the models

```{r}
plot(decision_tree)
```

```{r}
plot(mod_rf)
```

```{r}
plot(mod_gbm)
```

