library(data.table)

library(laGP)
library(R.matlab)
# library(xgboost)
# library(FNN)
# library(glmnet)
# library(ranger)
# library(e1071)
#
# library(baeirGPR)
#####
rmse <- function(y_hat, y, method = "") {
  out <- sqrt(mean((y - y_hat)^2))
  cat(method,"rmse = ", out, "\n")
  return(out)
}
#####

setwd('~/Desktop/gpr_testing/sarcos/')
#  The first 21 columns are the input variables, and the 22nd column is used as the target variable
data_train <- readMat('./sarcos_inv.mat')
data_test <- readMat('./sarcos_inv_test.mat')

ds_train <- data_train$sarcos.inv
ds_test <- data_test$sarcos.inv.test



















