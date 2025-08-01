# Q2c - Fit a logistic regression model for the probability of diverted US flights using as many features as possible.
# Visualize the coefficients across years.
install.packages("mlr3")
install.packages("mlr3viz")
install.packages("precrec")

library(dplyr)
library(tidyverse)
library(mlr3)
library(mlr3viz)

#2004 
sapply(df_2004, class) # cannot use chr data type, so convert to factor type 
df_2004 <- df_2004 %>% 
  mutate(UniqueCarrier = as.factor(UniqueCarrier),
         TailNum = as.factor(TailNum),
         Origin = as.factor(Origin),
         Dest = as.factor(Dest),
         CancellationCode = as.factor(CancellationCode),
         Diverted = as.factor(Diverted))
  
# TaskClassif
# Specifying the $positive class, ensuring 1 = Yes, 0 = No 
task_classif_04 = as_task_classif(df_2004, target = "Diverted", positive = "1")
task_classif_04$positive 

# partitioning the data 
set.seed(1) # reproducibility 
splits = partition(task_classif_04) # match splits with Python
splits

# predicting posterior probabilites 
lrn_rpart = lrn("classif.rpart", predict_type = "prob")
lrn_rpart$train(task_classif_04, row_ids = splits$train)
prediction_2004 = lrn_rpart$predict(task_classif_04, row_ids = splits$test)
print(prediction_2004)

prob_diverted_04 <- prediction_2004$prob[, "1"]
prob_diverted_04_mean <- mean(prob_diverted_04)
# print(prob_diverted_04)

#visualisation ROC curve 

autoplot(prediction_2004, type = "roc")
ggsave("roc_2004.png")

#2005
sapply(df_2005, class)
df_2005 <- df_2005 %>% 
  mutate(UniqueCarrier = as.factor(UniqueCarrier),
         TailNum = as.factor(TailNum),
         Origin = as.factor(Origin),
         Dest = as.factor(Dest),
         CancellationCode = as.factor(CancellationCode),
         Diverted = as.factor(Diverted))

task_classif_05 = as_task_classif(df_2005, target = "Diverted", positive = "1")
task_classif_05$positive 

set.seed(2) # reproducibility 
splits = partition(task_classif_05)

lrn_rpart = lrn("classif.rpart", predict_type = "prob")
lrn_rpart$train(task_classif_05, splits$train)
prediction_2005 = lrn_rpart$predict(task_classif_05, splits$test)
print(prediction_2005)

prob_diverted_05 <- prediction_2005$prob[, "1"]
prob_diverted_05_mean <- mean(prob_diverted_05)

autoplot(prediction_2005, type = "roc")
ggsave("roc_2005.png")

#2006
sapply(df_2006, class)
df_2006 <- df_2006 %>% 
  mutate(UniqueCarrier = as.factor(UniqueCarrier),
         TailNum = as.factor(TailNum),
         Origin = as.factor(Origin),
         Dest = as.factor(Dest),
         CancellationCode = as.factor(CancellationCode),
         Diverted = as.factor(Diverted))
         
task_classif_06 = as_task_classif(df_2006, target = "Diverted", positive = "1")
task_classif_06$positive 

set.seed(3) # reproducibility 
splits = partition(task_classif_06)

lrn_rpart = lrn("classif.rpart", predict_type = "prob")
lrn_rpart$train(task_classif_06, splits$train)
prediction_2006 = lrn_rpart$predict(task_classif_06, splits$test)
print(prediction_2006)

prob_diverted_06 <- prediction_2006$prob[, "1"]
prob_diverted_06_mean <- mean(prob_diverted_06)

autoplot(prediction_2006, type = "roc")
ggsave("roc_2006.png")

#2007
sapply(df_2007, class)
df_2007 <- df_2007 %>% 
  mutate(UniqueCarrier = as.factor(UniqueCarrier),
         TailNum = as.factor(TailNum),
         Origin = as.factor(Origin),
         Dest = as.factor(Dest),
         CancellationCode = as.factor(CancellationCode),
         Diverted = as.factor(Diverted))
         
task_classif_07 = as_task_classif(df_2007, target = "Diverted", positive = "1")
task_classif_07$positive 

set.seed(4) # reproducibility 
splits = partition(task_classif_07)

lrn_rpart = lrn("classif.rpart", predict_type = "prob")
lrn_rpart$train(task_classif_07, splits$train)
prediction_2007 = lrn_rpart$predict(task_classif_07, splits$test)
print(prediction_2007)

prob_diverted_07 <- prediction_2007$prob[, "1"]
prob_diverted_07_mean <- mean(prob_diverted_07)

autoplot(prediction_2007, type = "roc")
ggsave("roc_2007.png")

#2008
sapply(df_2008, class)
df_2008 <- df_2008 %>% 
  mutate(UniqueCarrier = as.factor(UniqueCarrier),
         TailNum = as.factor(TailNum),
         Origin = as.factor(Origin),
         Dest = as.factor(Dest),
         CancellationCode = as.factor(CancellationCode),
         Diverted = as.factor(Diverted))
         
task_classif_08 = as_task_classif(df_2008, target = "Diverted", positive = "1")
task_classif_08$positive 

set.seed(5) # reproducibility 
splits = partition(task_classif_08)

lrn_rpart = lrn("classif.rpart", predict_type = "prob")
lrn_rpart$train(task_classif_08, splits$train)
prediction_2008 = lrn_rpart$predict(task_classif_08, splits$test)
print(prediction_2008)

prob_diverted_08 <- prediction_2008$prob[, "1"]
prob_diverted_08_mean <- mean(prob_diverted_08)

autoplot(prediction_2008, type = "roc")
ggsave("roc_2008.png")

# Diversion probability - avg predited probability based on all flights 
print(prob_diverted_04_mean)
print(prob_diverted_05_mean)
print(prob_diverted_06_mean)
print(prob_diverted_07_mean)
print(prob_diverted_08_mean)




