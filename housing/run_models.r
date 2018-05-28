library(data.table)

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
source('~/Desktop/baeirGPR/R/call_by_user.R')
Rcpp::sourceCpp('~/Desktop/baeirGPR/src/matprod.cpp')
source('~/Desktop/baeirGPR/R/gpr.R')
source('~/Desktop/baeirGPR/R/local_gpr.R')
source('~/Desktop/baeirGPR/R/gpr_tuning.R')
source('~/Desktop/baeirGPR/R/boosting_gpr.R')
#####
setwd("~/Desktop/gpr_testing/housing")
ds <- fread("./housing.data")

feature_names <- c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
                   "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV")
ds <- setNames(ds, feature_names)
write.csv(ds, file = "housing.csv")


ds <- as.data.frame(apply(ds, 2, as.numeric))

ndata <- nrow(ds)
test_size <- floor(ndata/10)

set.seed(8787)
test_idx <- sample(ndata, test_size)

ds_train <- ds[-test_idx,]
ds_train_x <- model.matrix(MEDV~.-1, ds_train)
ds_train_y <- as.matrix(ds_train$MEDV)
ds_test <- ds[test_idx,]
ds_test_x <- model.matrix(MEDV~.-1, ds_test)
ds_test_y <- as.matrix(ds_test$MEDV)
#######

## ds2 is for baeirGPR
ds2 <- ds
# mean of y shift
mean_shift = mean(ds_train_y)
ds2$MEDV <- ds2$MEDV - mean(ds_train_y)
# normalize features
for (ii in 1:ncol(ds_train_x) ) {
  m_i <- mean(ds_train[,ii])
  std_i <- sd(ds_train[,ii])
  ds2[,ii] <- (ds2[,ii] - m_i)/std_i
}

ds2_train <- ds2[-test_idx,]
ds2_train_x <- model.matrix(MEDV~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$MEDV)
ds2_test <- ds2[test_idx,]
ds2_test_x <- model.matrix(MEDV~.-1, ds2_test)
ds2_test_y <- as.matrix(ds2_test$MEDV)

# try baeirGPR
# t_size <- nrow(ds2_train_x)
# t_idx <- sample(nrow(ds_train), t_size)
# ds_tune_x <- ds2_train_x[t_idx,]
# ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param1 = gpr_tune(ds2_train_x, ds2_train_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
kern_param1 = gpr_tune(ds2_test_x, ds2_test_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50, init_betainv = kern_param1$betainv,
                       init_theta = kern_param1$thetarel)
saveRDS(kern_param1, 'kern_param1.rds')
# ======================
kern_param1 = readRDS('kern_param1.rds')
bsize = nrow(ds2_train)
nmodel = 300
update_k = 20
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> bsize_all_nmodel_200.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = 0.1, tune_param = TRUE,
                                       update_kparam_tiems = update_k, decay_lr = 0.9, update_lr = 40,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill -15 $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
t1
cat(" baeirGPR rmse =", tail(gbm_model1$test_rmse), '\n')
cat("Min baeirGPR (row)  rmse =", min(gbm_model1$test_rmse), '\n')

bsize = nrow(ds2_train)
nmodel = 300
update_k = 20
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> bsize_all_nmodel_200_gbm3.txt &'))
t2=system.time(gbm_model2 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "3",
                                       n_model = nmodel, batch_size = bsize, lr = 0.01, tune_param = FALSE,
                                       update_kparam_tiems = update_k,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill -15 $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
t1
cat(" baeirGPR rmse =", tail(gbm_model1$test_rmse))
cat("Min baeirGPR (col)  rmse =", min(gbm_model1$test_rmse))

bsize = 225
nmodel = 300
update_k = 20
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> sr_bsize_225_nmodel_200.txt &'))
t3=system.time(gbm_model3 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "gbm_sr",
                                       n_model = nmodel, batch_size = bsize, lr = 0.01, tune_param = TRUE,
                                       update_kparam_tiems = update_k, sr_size = 10,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
t1
cat(" baeirGPR rmse =", tail(gbm_model1$test_rmse))
cat("Min baeirGPR (sr)  rmse =", min(gbm_model1$test_rmse))

# try ols
lm.fit <- lm(MEDV~.,data = ds_train)
pred_ols <- predict(lm.fit, newdata=ds_test)
rmse_ols <- rmse(pred_ols, ds_test_y, "ols")

# try xgboost
n_feature = ncol(ds_train_x)
tmp_xgb <- xgboost(data = data.matrix(ds_train_x),
                   label = ds_train_y,
                   eta = 0.025,
                   max_depth = 6,
                   nround = 750,
                   subsample = 0.7,
                   nthread = 4,
                   colsample_bytree = 1,
                   seed = 1
)
pred_xgb <- predict(tmp_xgb, ds_test_x)
rmse_xgb <- rmse(pred_xgb, ds_test_y, "xgboost")

# full gpr
gpr_model1 <- traintraintrain(ds2_train_x, ds2_train_y, pred_method = "cg_direct_lm",
                              kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                              kbetainv = kern_param1$betainv, ncpu = -1, srsize = NULL,
                              clus_size = NULL)
gpr_pred1 <- gpr_fit(ds2_test_x, ds2_train_x, gpr_model1)
rmse_fullGPR <- rmse(gpr_pred1, ds2_test_y, "full gpr")

# try kNN
pred_knn1 <- knn.reg(ds_train_x, ds_test_x, ds_train_y, k = 8)$pred
rmse_knn <- rmse(pred_knn1, ds_test_y, "knn")

# try LASSO
mdl_lasso = cv.glmnet(ds_train_x, ds_train_y, family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = ds_test_x)
rmse(pred_lasso, ds_test_y, "LASSO")

# try RIDGE
lambdas <- 10^seq(40, -50, by = -.1)
mdl_ridge = cv.glmnet(ds_train_x, ds_train_y, family = "gaussian", alpha = 0, lambda = lambdas)
pred_ridge = predict(mdl_ridge, newx = ds_test_x)
rmse(pred_ridge, ds_test_y, "RIDGE")

# try random forest
mdl_rf = ranger(MEDV ~ ., data = ds_train, num.trees = 10, mtry = 5, write.forest = T)
pred_rf = predict(mdl_rf, ds_test_x)
rmse_rf = rmse(pred_rf$predictions, ds_test_y, "random forest")

# try svr
#linear kernel
mdl_svr = svm(MEDV~., ds2_train, kernel = "linear")
svr.pred = predict(mdl_svr, ds2_test_x)
rmse(svr.pred, ds2_test_y, "svr")
#polynomial kernel
mdl_svr = svm(MEDV~., ds2_train, kernel = "polynomial", degree = 3)
svr.pred = predict(mdl_svr, ds2_test_x)
rmse(svr.pred, ds2_test_y, "svr")

# rbf kernel
mdl_svr = tune.svm(MEDV~., data = ds2_train, kernel = "radial", gamma = 2^c(-10:-2), cost = 2^c(-2:6))
svr.pred = predict(mdl_svr$best.model, ds2_test_x)
rmse(svr.pred, ds2_test_y, "svr")
#
tuneResult <- tune(svm, MEDV~.,  data = ds2_train,
                   ranges = list(epsilon = seq(0,1,0.01), cost = 2^(2:9))
)
tunedModel <- tuneResult$best.model
tunedModelY <- predict(tunedModel, ds2_test_x)
rmse(tunedModelY, ds2_test_y, "svr tune")

