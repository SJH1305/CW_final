# Q2c - Fit a logistic regression model for the probability of diverted US flights using as many features as possible.
# Visualize the coefficients across years.
install.packages("mlr3")
install.packages("mlr3viz")
install.packages("precrec")
install.packages("mlr3learners")
install.packages("mlr3pipelines")

library(dplyr)
library(tidyverse)
library(mlr3)
library(mlr3viz)
library(mlr3learners)
library(mlr3pipelines)
library(ggplot2)

#2004 
# remove features from df itself 
df_2004_ml <- df_2004 %>% 
  select(-ActualElapsedTime, -Cancelled, -CancellationCode, -DepTime, -ArrTime, -AirTime, -ArrDelay, -DepDelay,
        -CarrierDelay, -WeatherDelay,-NASDelay, -SecurityDelay, -LateAircraftDelay, -TaxiIn,-TaxiOut)


# cannot use chr data type, so convert to factor type 
df_2004_ml <- df_2004_ml %>% 
  mutate(UniqueCarrier = as.factor(UniqueCarrier),
         FlightNum = as.factor(FlightNum), 
         TailNum = as.factor(TailNum),
         Origin = as.factor(Origin),
         Dest = as.factor(Dest),
         Month = as.numeric(Month),
         DayofMonth = as.numeric(DayofMonth),
         DayOfWeek = as.numeric(DayOfWeek),
         CRSDepTime = as.numeric(CRSDepTime),
         CRSArrTime = as.numeric(CRSArrTime),
         Distance = as.numeric(Distance),
         Year = as.numeric(Year),
         CRSElapsedTime = as.numeric(CRSElapsedTime),
         Diverted = as.factor(Diverted))
         
# TaskClassif
# Specifying the $positive class, ensuring 1 = Yes, 0 = No 
task_2004 <- TaskClassif$new(id = "diverted_2004",
                             backend = df_2004_ml,
                             target = "Diverted",
                             positive = "1")

# partitioning the data 
set.seed(1) # reproducibility 
splits <- partition(task_2004) 
train <- splits$train
test <- splits$test

# imputation - helps with NaNs 
imputer <- po("imputemean") %>>% po("imputemode")  # mean and mode for categorical and numerical like in python ver.

# predicting posterior probabilities 
learner <- lrn("classif.log_reg", predict_type = "prob")
learner

combined <- imputer %>>% learner # note different pipe operator here and for imputer
combined_learner <- GraphLearner$new(combined)

# Learner training 
combined_learner$train(task_2004, row_ids = train)
prediction_2004 <- combined_learner$predict(task_2004, row_ids = test)

#visualisation ROC curve 
autoplot(prediction_2004, type = "roc")
ggsave("roc_2004.png")

#2005
df_2005_ml <- df_2005 %>% 
  select(-ActualElapsedTime, -Cancelled, -CancellationCode, -DepTime, -ArrTime, -AirTime, -ArrDelay, -DepDelay,
         -CarrierDelay, -WeatherDelay,-NASDelay, -SecurityDelay, -LateAircraftDelay, -TaxiIn,-TaxiOut)

df_2005_ml <- df_2005_ml %>% 
  mutate(UniqueCarrier = as.factor(UniqueCarrier),
         FlightNum = as.factor(FlightNum), 
         TailNum = as.factor(TailNum),
         Origin = as.factor(Origin),
         Dest = as.factor(Dest),
         Month = as.numeric(Month),
         DayofMonth = as.numeric(DayofMonth),
         DayOfWeek = as.numeric(DayOfWeek),
         CRSDepTime = as.numeric(CRSDepTime),
         CRSArrTime = as.numeric(CRSArrTime),
         Distance = as.numeric(Distance),
         Year = as.numeric(Year),
         CRSElapsedTime = as.numeric(CRSElapsedTime),
         Diverted = as.factor(Diverted))

task_2005 <- TaskClassif$new(id = "diverted_2005",
                             backend = df_2005_ml,
                             target = "Diverted",
                             positive = "1")

set.seed(1) # reproducibility 
splits <- partition(task_2005) 
train <- splits$train
test <- splits$test

imputer <- po("imputemean") %>>% po("imputemode")  # mean and mode for categorical and numerical like in python ver.

learner <- lrn("classif.log_reg", predict_type = "prob")
learner

combined <- imputer %>>% learner # note different pipe operator here and for imputer
combined_learner <- GraphLearner$new(combined)

combined_learner$train(task_2005, row_ids = train)
prediction_2005 <- combined_learner$predict(task_2005, row_ids = test)

autoplot(prediction_2005, type = "roc")
ggsave("roc_2005.png")

#2006
df_2006_ml <- df_2006 %>% 
  select(-ActualElapsedTime, -Cancelled, -CancellationCode, -DepTime, -ArrTime, -AirTime, -ArrDelay, -DepDelay,
         -CarrierDelay, -WeatherDelay,-NASDelay, -SecurityDelay, -LateAircraftDelay, -TaxiIn,-TaxiOut)


# cannot use chr data type, so convert to factor type 
df_2006_ml <- df_2006_ml %>% 
  mutate(UniqueCarrier = as.factor(UniqueCarrier),
         FlightNum = as.factor(FlightNum), 
         TailNum = as.factor(TailNum),
         Origin = as.factor(Origin),
         Dest = as.factor(Dest),
         Month = as.numeric(Month),
         DayofMonth = as.numeric(DayofMonth),
         DayOfWeek = as.numeric(DayOfWeek),
         CRSDepTime = as.numeric(CRSDepTime),
         CRSArrTime = as.numeric(CRSArrTime),
         Distance = as.numeric(Distance),
         Year = as.numeric(Year),
         CRSElapsedTime = as.numeric(CRSElapsedTime),
         Diverted = as.factor(Diverted))

# TaskClassif
# Specifying the $positive class, ensuring 1 = Yes, 0 = No 
task_2006 <- TaskClassif$new(id = "diverted_2006",
                             backend = df_2006_ml,
                             target = "Diverted",
                             positive = "1")

# partitioning the data 
set.seed(1) # reproducibility 
splits <- partition(task_2006) 
train <- splits$train
test <- splits$test

# imputation - helps with NaNs 
imputer <- po("imputemean") %>>% po("imputemode")  # mean and mode for categorical and numerical like in python ver.

# predicting posterior probabilities 
learner <- lrn("classif.log_reg", predict_type = "prob")
learner

combined <- imputer %>>% learner # note different pipe operator here and for imputer
combined_learner <- GraphLearner$new(combined)

# Learner training 
combined_learner$train(task_2006, row_ids = train)
prediction_2006 <- combined_learner$predict(task_2006, row_ids = test)

autoplot(prediction_2006, type = "roc")
ggsave("roc_2006.png")

#2007
df_2007_ml <- df_2007 %>% 
  select(-ActualElapsedTime, -Cancelled, -CancellationCode, -DepTime, -ArrTime, -AirTime, -ArrDelay, -DepDelay,
         -CarrierDelay, -WeatherDelay,-NASDelay, -SecurityDelay, -LateAircraftDelay, -TaxiIn,-TaxiOut)


# cannot use chr data type, so convert to factor type 
df_2007_ml <- df_2007_ml %>% 
  mutate(UniqueCarrier = as.factor(UniqueCarrier),
         FlightNum = as.factor(FlightNum), 
         TailNum = as.factor(TailNum),
         Origin = as.factor(Origin),
         Dest = as.factor(Dest),
         Month = as.numeric(Month),
         DayofMonth = as.numeric(DayofMonth),
         DayOfWeek = as.numeric(DayOfWeek),
         CRSDepTime = as.numeric(CRSDepTime),
         CRSArrTime = as.numeric(CRSArrTime),
         Distance = as.numeric(Distance),
         Year = as.numeric(Year),
         CRSElapsedTime = as.numeric(CRSElapsedTime),
         Diverted = as.factor(Diverted))

# TaskClassif
# Specifying the $positive class, ensuring 1 = Yes, 0 = No 
task_2007 <- TaskClassif$new(id = "diverted_2007",
                             backend = df_2007_ml,
                             target = "Diverted",
                             positive = "1")

# partitioning the data 
set.seed(1) # reproducibility 
splits <- partition(task_2007) 
train <- splits$train
test <- splits$test

# imputation - helps with NaNs 
imputer <- po("imputemean") %>>% po("imputemode")  # mean and mode for categorical and numerical like in python ver.

# predicting posterior probabilities 
learner <- lrn("classif.log_reg", predict_type = "prob")
learner

combined <- imputer %>>% learner # note different pipe operator here and for imputer
combined_learner <- GraphLearner$new(combined)

# Learner training 
combined_learner$train(task_2007, row_ids = train)
prediction_2007 <- combined_learner$predict(task_2007, row_ids = test)

autoplot(prediction_2007, type = "roc")
ggsave("roc_2007.png")

#2008
df_2008_ml <- df_2008 %>% 
  select(-ActualElapsedTime, -Cancelled, -CancellationCode, -DepTime, -ArrTime, -AirTime, -ArrDelay, -DepDelay,
         -CarrierDelay, -WeatherDelay,-NASDelay, -SecurityDelay, -LateAircraftDelay, -TaxiIn,-TaxiOut)


# cannot use chr data type, so convert to factor type 
df_2008_ml <- df_2008_ml %>% 
  mutate(UniqueCarrier = as.factor(UniqueCarrier),
         FlightNum = as.factor(FlightNum), 
         TailNum = as.factor(TailNum),
         Origin = as.factor(Origin),
         Dest = as.factor(Dest),
         Month = as.numeric(Month),
         DayofMonth = as.numeric(DayofMonth),
         DayOfWeek = as.numeric(DayOfWeek),
         CRSDepTime = as.numeric(CRSDepTime),
         CRSArrTime = as.numeric(CRSArrTime),
         Distance = as.numeric(Distance),
         Year = as.numeric(Year),
         CRSElapsedTime = as.numeric(CRSElapsedTime),
         Diverted = as.factor(Diverted))

# TaskClassif
# Specifying the $positive class, ensuring 1 = Yes, 0 = No 
task_2008 <- TaskClassif$new(id = "diverted_2008",
                             backend = df_2008_ml,
                             target = "Diverted",
                             positive = "1")

# partitioning the data 
set.seed(1) # reproducibility 
splits <- partition(task_2008) 
train <- splits$train
test <- splits$test

# imputation - helps with NaNs 
imputer <- po("imputemean") %>>% po("imputemode")  # mean and mode for categorical and numerical like in python ver.

# predicting posterior probabilities 
learner <- lrn("classif.log_reg", predict_type = "prob")
learner

combined <- imputer %>>% learner # note different pipe operator here and for imputer
combined_learner <- GraphLearner$new(combined)

# Learner training 
combined_learner$train(task_2008, row_ids = train)
prediction_2008 <- combined_learner$predict(task_2008, row_ids = test)

autoplot(prediction_2008, type = "roc")
ggsave("roc_2008.png")




