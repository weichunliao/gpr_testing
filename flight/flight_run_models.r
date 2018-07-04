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
# source('~/Desktop/baeirGPR/R/call_by_user.R')
# Rcpp::sourceCpp('~/Desktop/baeirGPR/src/matprod.cpp')
# source('~/Desktop/baeirGPR/R/gpr.R')
# source('~/Desktop/baeirGPR/R/local_gpr.R')
# source('~/Desktop/baeirGPR/R/gpr_tuning.R')
# source('~/Desktop/baeirGPR/R/boosting_gpr.R')
#####
setwd("~/Desktop/gpr_testing/flight/")

tmp = fread('./processed_ttest.txt')
tmp$V1 = NULL
tmp$DEPARTURE_DELAY = NULL
ds = as.data.frame(tmp)

# step0. select column
# used_col = c("ARRIVAL_DELAY", "MONTH", "DAY_OF_WEEK", "DISTANCE", "SCHEDULED_HOUR",
#              "AA", "AS", "B6", "DL", "EV", "F9", "HA", "MQ", "NK", "OO", "UA",
#              "US", "VX", "WN")
# random permutation
set.seed(2018)
# ds = ds[,c(2:36)]
ds <- ds[sample(nrow(ds)),]

# step1. set testing size
test_size = floor(nrow(ds)/5)
test_idx = sample(nrow(ds), test_size)
ds_train = ds[-test_idx,]
ds_test = ds[test_idx,]
ds_test_y = as.matrix(ds_test$ARRIVAL_DELAY)

# step2. shift mean to zero
ds2 = ds
##### ver.1
# ds2$ARRIVAL_DELAY = log(ds2$ARRIVAL_DELAY+88)
# mean_shift = mean(log(ds_train$ARRIVAL_DELAY+88))
# ds2$ARRIVAL_DELAY = ds2$ARRIVAL_DELAY - mean_shift
##### ver.2
mean_shift = mean(ds_train$ARRIVAL_DELAY)
ds2$ARRIVAL_DELAY = ds2$ARRIVAL_DELAY - mean_shift

# step3. feature normailize (distance)
m1 = mean(ds_train$DISTANCE)
sd1 = sd(ds_train$DISTANCE)
ds2$DISTANCE = (ds2$DISTANCE - m1)/sd1

# step4. train-test split
ds2_train = ds2[-test_idx,]
ds2_test = ds2[test_idx,]

ds2_train_x = model.matrix(ARRIVAL_DELAY~.-1, ds2_train)
ds2_train_y = as.matrix(ds2_train$ARRIVAL_DELAY)
ds2_test_x = model.matrix(ARRIVAL_DELAY~.-1, ds2_test)
ds2_test_y = as.matrix(ds2_test$ARRIVAL_DELAY)

# step5. tuning kernel parameters
### kparam with ARD
t_size <- 1000
t_idx <- sample(nrow(ds2_train_x), t_size)
ds_tune_x <- ds2_train_x[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param2 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
t_idx <- sample(nrow(ds2_train_x), t_size)
ds_tune_x <- ds2_trainmx[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param2 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50, init_betainv = kern_param2$betainv,
                       init_theta = kern_param2$thetarel)
saveRDS(kern_param2, "kern_param2.rds")
### kparam without ARD
t_size <- 1000
t_idx <- sample(nrow(ds2_train_x), t_size)
ds_tune_x <- ds2_train_x[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param2n = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = F, optim_ard_max = 50)
t_idx <- sample(nrow(ds2_train_x), t_size)
ds_tune_x <- ds2_trainmx[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param2n = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = F, optim_ard_max = 50, init_betainv = kern_param2n$betainv,
                       init_theta = kern_param2n$thetarel)
saveRDS(kern_param2n, "kern_param2n.rds")

# step6. run gbm models


#####################
# ols
lm.fit = lm(ARRIVAL_DELAY~.,data = ds2_train)
pred_ols = predict(lm.fit, newdata = as.data.frame(ds2_test_x))
# pred_ols = exp(pred_ols+mean_shift)-88
rmse_ols <- rmse(pred_ols, ds2_test_y, "ols")
# rmse_ols <- rmse(pred_ols, ds_test_y, "ols")

# xgboost
tmp_xgb <- xgboost(data = data.matrix(ds2_train_x),
                   label = ds2_train_y,
                   eta = 0.3,
                   max_depth = 5,
                   nround = 300,
                   subsample = 0.5,
                   colsample_bytree = 1,
                   seed = 1
)
pred_xgb <- predict(tmp_xgb, data.matrix(ds2_test_x))
# pred_xgb = exp(pred_xgb+mean_shift)-88
rmse_xgb <- rmse(pred_xgb, ds2_test_y, "xgboost")

# try kNN
pred_knn <- knn.reg(ds2_train_x, ds2_test_x, ds2_train_y, k = 5)$pred
# pred_knn = exp(pred_knn+mean_shift) -88
rmse_knn <- rmse(pred_knn, ds2_test_y, "knn")

# try LASSO
mdl_lasso = cv.glmnet(ds2_train_x, ds2_train_y, family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = ds2_test_x)
# pred_lasso = exp(pred_lasso+mean_shift)-88
rmse(pred_lasso, ds2_test_y, "LASSO")

# try RIDGE
lambdas <- 10^seq(3, -3, by = -0.1)
mdl_ridge = cv.glmnet(data.matrix(ds2_train_x), data.matrix(ds2_train_y),
                      family = "gaussian", alpha = 0, lambda = lambdas)
pred_ridge = predict(mdl_ridge, newx = data.matrix(ds2_test_x))
# pred_ridge = exp(pred_ridge+mean_shift)-88
rmse(pred_ridge, ds2_test_y, "RIDGE")

# try random forest
mdl_rf = ranger(yy ~ ., data = ds2_train, num.trees = 100, mtry = 12, write.forest = T)
pred_rf = predict(mdl_rf, ds2_test)
pred_rf2 = pred_rf$predictions
# pred_rf2 = exp(pred_rf2+mean_shift) -88
rmse(pred_rf2, ds2_test_y, "random forest")

# try svr ###########################
#  linear
time.linear = system.time(mdl_svr <- svm(yy~., ds2_train, kernel = "linear"))
svr.pred = predict(mdl_svr, ds2_test_x)
# svr.pred = exp(svr.pred+mean_shift) -88
rmse_svr = rmse(svr.pred, ds2_test_y, "svr linear")

# poly
time.poly = system.time(mdl_svr <- svm(yy~., ds2_train, kernel = "polynomial", degree = 3))
svr.pred = predict(mdl_svr, ds2_test_x)
# svr.pred = exp(svr.pred+mean_shift) -88
rmse_svr = rmse(svr.pred, ds2_test_y, "svr poly")

# rbf
time.rbf = system.time(mdl_svr <- tune.svm(yy~., data = ds2_train, kernel = "radial", gamma = 2^c(-8:0), cost = 2^c(-4:4)))
svr.pred = predict(mdl_svr$best.model, ds2_test_x)
# svr.pred = exp(svr.pred+mean_shift) -88
rmse_svr = rmse(svr.pred, ds2_test_y, "svr rbf")



