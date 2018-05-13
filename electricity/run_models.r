library(data.table)

library(xgboost)
library(FNN)
library(glmnet)
library(ranger)
#####
rmse <- function(y_hat, y, method = "") {
  out <- sqrt(mean((y - y_hat)^2))
  cat(method,"rmse = ", out, "\n")
  return(out)
}
#####
setwd("~/Desktop/gpr_testing/electricity")
ds <- fread("~/Desktop/gpr_testing/electricity/LD2011_2014.txt")


