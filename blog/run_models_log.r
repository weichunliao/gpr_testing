library(data.table)
library(caret)

library(xgboost)
library(FNN)
library(glmnet)
library(ranger)
library(e1071)

library(baeirGPR)
#####
rmse <- function(y_hat, y, method = "") {
  out <- sqrt(mean((y - y_hat)^2))
  cat(method,"rmse = ", out, "\n")
  return(out)
}
#####

# task: predict the number of comments iin the next 24 hours
# V281: The target: the number of comments in the next 24 hours

# step0. load data and set col names
setwd("~/Desktop/gpr_testing/blog/")
ds_train <- fread("./BlogFeedback-Train2_log.csv")
ds_test <- fread("./BlogFeedback-Test2_log.csv")
ncol(ds_train)
setnames(ds_train, c(paste('x', 1:271, sep=''), 'y'))
setnames(ds_test, c(paste('x', 1:271, sep=''), 'y'))
ds_train = as.data.frame(ds_train)
ds_test = as.data.frame(ds_test)

origin_test_y = (exp(ds_test$y)-1)

#########################################################################
tmp_train <- as.data.frame(ds_train)
tmp_test <- as.data.frame(ds_test)

# step1. mean shift
# mean_shift = mean(tmp_train$y)
# tmp_train$y = tmp_train$y - mean_shift
# tmp_test$y = tmp_test$y - mean_shift

# step2. normalize continuous var
for (i in c(1:54, 270, 271)){
  m_train <- mean(tmp_train[,i])
  std_train <- sd(tmp_train[,i])

  tmp_train[,i] <- (tmp_train[,i] - m_train)/std_train
  tmp_test[,i] <- (tmp_test[,i] - m_train)/std_train
}

# step3. remove linear dependent col
# library(caret)
# rm_col_idx = findLinearCombos(tmp_train[,-272])$remove
rm_col_idx = c(16,  47,  52,  65,  68,  72,  73,  86,  89, 122, 124, 141, 148, 153, 158, 161, 164, 171,
               182, 190, 192, 204, 228, 235, 242 ,268)
ds2_train = tmp_train[, -rm_col_idx]
ds2_test = tmp_test[, -rm_col_idx]
ncol(ds2_train)

ds_train = ds_train[, -rm_col_idx]
ds_test = ds_test[, -rm_col_idx]

# step.4 model matrix
ds2_train_x <- model.matrix(y~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$y)

ds2_test_x <- model.matrix(y~.-1, ds2_test)
ds2_test_y <- as.matrix(ds2_test$y)



### baeirGPR ####
# step.5 tune kern_param1_log.rds


# step.6 run gbm models


#################
# ols
lm.fit = lm(y~.,data = ds2_train)
pred_ols = predict(lm.fit, newdata = as.data.frame(ds2_test_x))
pred_ols = exp(pred_ols) - 1
rmse_ols <- rmse(pred_ols, origin_test_y, "ols")















