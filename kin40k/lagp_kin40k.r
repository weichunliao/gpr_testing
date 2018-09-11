library(data.table)

library(laGP)
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

setwd('~/Desktop/gpr_testing/kin40k/')
ds_train_x = as.matrix(fread('./kin40k_train_data.asc'))
ds_train_y = as.matrix(fread('./kin40k_train_labels.asc'))
ds_test_x = as.matrix(fread('./kin40k_test_data.asc'))
ds_test_y = as.matrix(fread('./kin40k_test_labels.asc'))

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

# X = head(ds_train_x, 50)
X = ds_train_x
# Y = head(ds_train_y, 50)
Y = ds_train_y
# Xref = head(ds_test_x)
Xref = ds_test_x

################ mspe #######################
t0 = Sys.time()
p.mspe = laGP(Xref, 100, 200, X, Y, method="mspe")
t1 = Sys.time()
print(t1-t0)
rmse(p.mspe$mean, ds_test_y, "p.mspe")

########## alc ###################
t0 = Sys.time()
p.alc = laGP(Xref, 100, 200, X, Y, method="alc")
t1 = Sys.time()
print(t1-t0)

rmse(p.alc$mean, ds_test_y, "alc")




