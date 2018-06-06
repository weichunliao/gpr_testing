library(data.table)
library(dummies)

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
setwd('~/Desktop/gpr_testing/e2006log/')

ds_train = read.csv('./log1p.E2006.train')
ds_test = read.csv('./log1p.E2006.test')






