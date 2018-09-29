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

# tune kernel parameter
t_size <- 1000
t_idx <- sample(nrow(ds2_train_x), t_size)
ds_tune_x <- ds2_train_x[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param1 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
t_idx <- sample(nrow(ds2_train_x), t_size)
ds_tune_x <- ds2_train_x[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param1 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50, init_betainv = kern_param1$betainv,
                       init_theta = kern_param1$thetarel)
saveRDS(kern_param1, "kern_param1.rds")
############
t_size <- 1000
t_idx <- sample(nrow(ds2_train_x), t_size)
ds_tune_x <- ds2_train_x[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param1n = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                        optim_report = 5, optim_ard_report = 5,
                        ARD = F, optim_ard_max = 50)
t_idx <- sample(nrow(ds2_train_x), t_size)
ds_tune_x <- ds2_train_x[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param1n = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                        optim_report = 5, optim_ard_report = 5,
                        ARD = F, optim_ard_max = 50, init_betainv = kern_param1n$betainv,
                        init_theta = kern_param1n$thetarel)
saveRDS(kern_param1n, "kern_param1n.rds")

# run gpr
kern_param1 = readRDS('kern_param1.rds')
bsize = 2000
nmodel = 700
update_k = 20
lr = 0.1
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> bsize_2000_nmodel_700.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, decay_lr = 0.9, update_lr = 40,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
# system(paste("kill -15 $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))

# gbm3
kern_param1n = readRDS('kern_param1n.rds')
bsize = 1000
nmodel = 700
update_k = 20
lr = 0.1
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> bsize_100_nmodel_700_gbm3.txt &'))
t2=system.time(gbm_model2 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "3",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = FALSE,
                                       update_kparam_tiems = update_k,
                                       kname = "gaussiandotrel", ktheta = kern_param1n$thetarel,
                                       kbetainv = kern_param1n$betainv))
# system(paste("kill -15 $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))

# gbm_sr
kern_param1 = readRDS('kern_param1.rds')
bsize = 6750
nmodel = 700
update_k = 20
lr = 0.05
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> sr_bsize_670_nmodel_700.txt &'))
t3=system.time(gbm_model3 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "gbm_sr",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, sr_size = 10,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))

# try ols
ds2_train = as.data.frame(cbind(ds2_train_x, ds2_train_y))
setnames(ds2_train, c(paste('V', 1:8, sep = ''), 'y'))
lm.fit <- lm(y~.,data = ds2_train)
pred_ols <- predict(lm.fit, newdata=as.data.frame(ds2_test_x))
rmse_ols <- rmse(pred_ols, ds2_test_y, "ols")

# try xgboost
n_feature = ncol(ds_train_x)
tmp_xgb <- xgboost(data = data.matrix(ds2_train_x),
                   label = ds2_train_y,
                   eta = 0.1,
                   max_depth = 6,
                   nround = 800,
                   subsample = 0.7,
                   nthread = 4,
                   colsample_bytree = 1,
                   seed = 1
)
pred_xgb <- predict(tmp_xgb, ds2_test_x)
rmse_xgb <- rmse(pred_xgb, ds2_test_y, "xgboost")

# try kNN
pred_knn1 <- knn.reg(ds2_train_x, ds2_test_x, ds_train_y, k = 8)$pred
rmse_knn <- rmse(pred_knn1, ds2_test_y, "knn")

# try LASSO
mdl_lasso = cv.glmnet(ds_train_x, ds_train_y, family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = ds_test_x)
rmse(pred_lasso, ds_test_y, "LASSO")

# try RIDGE
lambdas <- 10^seq(1, -1, by = -.001)
mdl_ridge = cv.glmnet(ds_train_x, ds_train_y, family = "gaussian", alpha = 0, lambda = lambdas)
pred_ridge = predict(mdl_ridge, newx = ds_test_x)
rmse(pred_ridge, ds_test_y, "RIDGE")

# try random forest
ds2_train = as.data.frame(cbind(ds2_train_x, ds2_train_y))
setnames(ds2_train, c(paste('V', 1:8, sep = ''), 'y'))
mdl_rf = ranger(y ~ ., data = ds2_train, num.trees = 100, mtry = 7, write.forest = T)
pred_rf = predict(mdl_rf, ds2_test_x)
rmse_rf = rmse(pred_rf$predictions, ds2_test_y, "random forest")

# try svr
#linear kernel
mdl_svr = svm(y~., ds2_train, kernel = "linear")
svr.pred = predict(mdl_svr, ds2_test_x)
rmse(svr.pred, ds2_test_y, "svr")

#polynomial kernel
mdl_svr = svm(y~., ds2_train, kernel = "polynomial", degree = 3)
svr.pred = predict(mdl_svr, ds2_test_x)
rmse(svr.pred, ds2_test_y, "svr")

# rbf kernel
ds2_train = as.data.frame(cbind(ds2_train_x, ds2_train_y))
setnames(ds2_train, c(paste('V', 1:8, sep = ''), 'y'))
t0 = Sys.time()
sub_train_idx = sample(nrow(ds2_train), 5000)
mdl_svr = tune.svm(y~., data = ds2_train[sub_train_idx,], kernel = "radial", gamma = 2^c(-10:-2), cost = 2^c(-2:6))
svr.pred = predict(mdl_svr$best.model, ds2_test_x)
t1 = Sys.time()
print(t1-t0)
rmse(svr.pred, ds2_test_y, "svr")
#
# tuneResult <- tune(svm, MEDV~.,  data = ds2_train,
#                    ranges = list(epsilon = seq(0,1,0.01), cost = 2^(2:9))
# )
# tunedModel <- tuneResult$best.model
# tunedModelY <- predict(tunedModel, ds2_test_x)
# rmse(tunedModelY, ds2_test_y, "svr tune")



