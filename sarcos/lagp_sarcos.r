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


ds_train_x = ds_train[,1:21]
ds_train_y = ds_train[,22]
ds_test_x = ds_test[,1:21]
ds_test_y = ds_test[,22]


# mean_shift + normalize
mean_shift_y = mean(ds_train_y)
ds2_train_y = ds_train_y - mean_shift_y
ds2_test_y = ds_test_y - mean_shift_y

ds2_train_x = ds_train_x
ds2_test_x = ds_test_x
for (i in c(1:ncol(ds_train_x))) {
  m1 = mean(ds_train_x[,i])
  s1 = sd(ds_train_x[,i])
  ds2_train_x[,i] = (ds2_train_x[,i] - m1)/s1
  ds2_test_x[,i] = (ds2_test_x[,i] - m1)/s1
}

###################################
# X = head(ds_train_x, 50)
X = ds2_train_x
# Y = head(ds_train_y, 50)
Y = ds2_train_y
# Xref = head(ds_test_x)
Xref = ds2_test_x


################ mspe #######################
t0 = Sys.time()
p.mspe = laGP(Xref, 900, 1000, X, Y, method="mspe")
t1 = Sys.time()
print(t1-t0)
rmse(p.mspe$mean, ds2_test_y, "p.mspe")

########## alc ###################
t0 = Sys.time()
p.alc = laGP(Xref, 100, 200, X, Y, method="alc")
t1 = Sys.time()
print(t1-t0)
rmse(p.alc$mean, ds2_test_y, "alc")
