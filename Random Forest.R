data <- read.csv(file.choose(), header = T)
str(data)
data$NSP <- as.factor(data$NSP)
table(data$NSP)


#Data Partition
set.seed(123)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7,0.3))
train <- data[ind==1,]
test <- data[ind==2,]


# #Random forest
# 1. RF developed by aggregating decision trees
# 2. Can be used for classification or regression
# 3. Avoids over fitting
# 4. Can deal with large number of features
# 5. Helps with feature selection based on importance
# 6. User-friendly: Depends on only 2 free parameters
# 6a. Number of Trees - ntree, default 500. It'll grow 500 decision trees based on which RF will be built
# 6b. Variables randomly selected as candidates at each split - mtry,
#     default is sq.root(p) for classification & p/3 for regression
#     p is number of features


######## STEPS
# 1. Draw ntree bootstrap samples
# 2. For each bootstrap sample, grow un-pruned tree by choosing best split based on
#    a random sample of mtry predictors at each node
# 3. Predict new data using majority votes for classification and avg for regression based on ntrees

# Random Forest
library(randomForest)
set.seed(222)
rf <- randomForest(NSP~., data=train)
print(rf)
attributes(rf)
rf$confusion

#Prediction and Confusion Matrix
library(caret)
p1 <- predict(rf, train)
table(p1)
head(p1)
head(data$NSP)

confusionMatrix(p1, train$NSP)

# Class: 1 Class: 2 Class: 3
# Sensitivity            1.0000   0.9950  1.00000  # how well we predicted diagonal / sum
# Specificity            0.9969   1.0000  1.00000
# Pos Pred Value         0.9991   1.0000  1.00000
# Neg Pred Value         1.0000   0.9992  1.00000
# Prevalence             0.7815   0.1377  0.08082
# Detection Rate         0.7815   0.1370  0.08082
# Detection Prevalence   0.7822   0.1370  0.08082
# Balanced Accuracy      0.9984   0.9975  1.00000

#predict test data
p2 <- predict(rf, test)
confusionMatrix(p2, test$NSP)


# Error rate
plot(rf)
#after 300 trees we are not able to lower the error rate
#hence tune the data


#Tune randomforest .... mtry
t <- tuneRF(train[,-22], train[,22],
       stepFactor = 0.5,
       plot = T,
       ntreeTry = 300,
       trace = T,
       improve = 0.05)

# #tuneRF(x,y, ntreeTry: numb of trees used at the tuning step,
#     stepFactor: mtry will decreaes by 0.5 on each iteration, 
#     improve: if there is an improvement of 0.05 then it will go ahead else it will stop)

rf1 <- randomForest(NSP~., data = train,
                    ntree=300, mtry=8,
                    importance = T, proximity=T)
rf1
rf

#Prediction and confusion matrix on train data
p1_rf1 <- predict(rf1, train)
confusionMatrix(p1_rf1, train$NSP)



#Prediction and confusion matrix on test data
p2_rf1 <- predict(rf1, test)
confusionMatrix(p2_rf1, test$NSP)

confusionMatrix(p2, test$NSP)


#Number of nodes for the trees
hist(treesize(rf),
     main="No. nodes for the trees",
     col = "green")


#variable importance
varImpPlot(rf1)

varImpPlot(rf1,
           sort = T,
           n.var = 10,
           main = "Top 10 - Var Imp Plot")

importance(rf1) #gives output in the form of a table
varUsed(rf1) #how many times a variable is used. the one which is used less will the not that important according to the plot

#Partial Dependence plot
partialPlot(rf1, train, ASTV, "1")
#for NSP=1 it predicts with high accuracy when ASTV is below 50

#Extract single tree
getTree(rf1, 1, labelVar = T)
#when status is -1 then that node is termial node and hence there will be some prediction


#multidimensinal Scaling plot of proximity matrix
MDSplot(rf1, train$NSP)




































































