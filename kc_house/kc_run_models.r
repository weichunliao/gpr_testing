library(data.table)

library(xgboost)
library(FNN)
library(glmnet)
library(ranger)

library(baeirGPR)
#########
# Rcpp::sourceCpp('~/Desktop/baeirGPR/src/matprod.cpp')
# source('~/Desktop/baeirGPR/R/gpr.R')
# source('~/Desktop/baeirGPR/R/local_gpr.R')
# source('~/Desktop/baeirGPR/R/gpr_tuning.R')
# source('~/Desktop/baeirGPR/R/boosting_gpr.R')
#####
rmse <- function(y_hat, y, method = "") {
  out <- sqrt(mean((y - y_hat)^2))
  cat(method,"rmse = ", out, "\n")
  return(out)
}
#####
setwd('~/Desktop/gpr_testing/kc_house/')

tmp = fread('./kc_house_data.csv')
tmp$id=NULL

