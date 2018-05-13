library(data.table)
library(lubridate)
library(dummies)

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
source('~/Desktop/baeirGPR/R/call_by_user.R')
Rcpp::sourceCpp('~/Desktop/baeirGPR/src/matprod.cpp')
source('~/Desktop/baeirGPR/R/gpr.R')
source('~/Desktop/baeirGPR/R/local_gpr.R')
source('~/Desktop/baeirGPR/R/gpr_tuning.R')
source('~/Desktop/baeirGPR/R/boosting_gpr.R')
#####
setwd("~/Desktop/gpr_testing/rtb")

# col.names <- c("id", "click", "hour", "C1", "banner_pos", "site_id", 
#                "site_domain", "site_category", "app_id", "app_domain",
#                "app_category", "device_id", "device_ip", "device_model",
#                "device_type", "device_conn_type", "C14", "C15", "C16", "C17",
#                "C18", "C19", "C20", "C21")
# col.classes <- c("character", "integer", "integer", rep("character", 21))
# 

# tmp = fread('~/Desktop/gpr_testing/rtb/train')
tmp_ds = readRDS('~/Desktop/gpr_testing/rtb/train.rds')

sub_size = nrow(tmp_ds)
mydata = tmp_ds[1:sub_size,]

lubridate::origin
mydata$weekday <- weekdays(as.Date(mydata$hour,origin="1970-01-01 UTC"))
mydata$time <- substr(mydata$hour,7,8)

feature_names = colnames(mydata)
# for (i in feature_names) {
#   cat(i, ': ', length(unique(mydata[[i]])), '\n')
#   
# }
mydata =as.data.frame(mydata)
used_names = c("click", "C1", "banner_pos", "site_category",
               "app_category", "device_type", "device_conn_type",
               "C15", "C16", "C18", "weekday", "time")

ss = 5000
ds_train = mydata[1:5000, used_names]

dummy_names <- c("C1", "site_category",
                 "app_category", "device_type", "device_conn_type",
                 "C15", "C16", "C18", "weekday")
ds2 <- dummy.data.frame(ds_train, names = dummy_names)
ds2 <- apply(ds2, 2, as.numeric)
ds2 <- as.data.frame(ds2)

for(ii in 2:(ncol(ds2))) {
  m1 <- mean(ds2[,ii])
  sd1 <- sd(ds2[,ii])
  ds2[,ii] <- (ds2[,ii] - m1)/sd1
}




