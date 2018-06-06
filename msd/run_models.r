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

# step1. load dataset
ds1 <- fread('./YearPredictionMSD.txt', header = FALSE)
load("./kern_gpr_param.rdata")
# ds1[1:5,1:5]

setnames(ds1, c('yy', paste('x', 1:90, sep = '')))
ds1 <- as.data.frame(ds1)

# normalize to mean 0 and unit variance.
mean_shift = mean(ds1$yy)
ds1$yy <- ds1$yy - mean(ds1$yy)
ds1_train = ds1[1:463715, ]
# ds1_test = ds1[463716:nrow(ds1),]

ds2 = ds1
for(ii in 2:ncol(ds1)) {
  m1 <- mean(ds1_train[,ii])
  sd1 <- sd(ds1_train[,ii])
  ds2[,ii] <- (ds2[,ii] - m1)/sd1
}

# step2. train-test split
# random permutation
# There are 463715 training records
ds2_train = ds2[1:463715, ]
ds2_test = ds2[463716:nrow(ds1),]
ds2_train <- ds2_train[sample(nrow(ds2_train)),]

# step2.1 set the size of training data
# ssize <- 460000
# ssize <- 8000
ssize <- 0
# > nrow(ds1)
# [1] 515345
# ds2 <- ds1[1:463715, ]
if(ssize > 0) {
  ds2_train <- ds2_train[1:ssize,]
}

# step2.2 matrix formulation
ds2_testmx <- model.matrix(yy~.-1, ds2_test)
ds2_test_y <- as.matrix(ds2_test$yy)
ds2_trainmx <- model.matrix(yy~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$yy)

# step3. tuning kernerl parameters
t_size <- 1000
t_idx <- sample(nrow(ds2_trainmx), t_size)
ds_tune_x <- ds2_trainmx[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param2 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
# ARD param (head)= 0.4911555 0.03532048 0.1225798 1e-06 0.4725913 0.2348435 1e-06 0.2105412 1e-06 1e-06
t_idx <- sample(nrow(ds2_trainmx), t_size)
ds_tune_x <- ds2_trainmx[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param2 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50, init_betainv = kern_param2$betainv,
                       init_theta = kern_param2$thetarel)
saveRDS(kern_param2, "kern_param2.rds")
############
t_size <- 1000
t_idx <- sample(nrow(ds2_trainmx), t_size)
ds_tune_x <- ds2_trainmx[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param2n = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                        optim_report = 5, optim_ard_report = 5,
                        ARD = F, optim_ard_max = 50)
# [NOARD] Optimal kernel parameter [theta0 theta1]= 134.8837 0.06912934
t_idx <- sample(nrow(ds2_trainmx), t_size)
ds_tune_x <- ds2_trainmx[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param2n = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                        optim_report = 5, optim_ard_report = 5,
                        ARD = F, optim_ard_max = 50, init_betainv = kern_param2n$betainv,
                        init_theta = kern_param2n$thetarel)
saveRDS(kern_param2n, "kern_param2n.rds")

# run gbm2
kern_param2 = readRDS('./kern_param2.rds')
bsize = 3000
nmodel = 1000
update_k = 50
lr = 0.3
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 60 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> gbm2_bsize4000_nmodel500.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_trainmx, ds2_train_y, ds2_testmx, ds2_test_y, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, decay_lr = 0.9, tune_size = 1000,
                                       kname = kern_param2$kernelname, ktheta = kern_param2$thetarel,
                                       kbetainv = kern_param2$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))

# run gbm3
kern_param2n = readRDS('./kern_param2n.rds')
bsize = 500
nmodel = 500
update_k = 50
lr = 0.4
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 60 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> gbm3_bsize2000_nmodel500.txt &'))
t2=system.time(gbm_model2 <- gbm_train(ds2_trainmx, ds2_train_y, ds2_testmx, ds2_test_y, pred_method = "3",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = FALSE,
                                       update_kparam_tiems = update_k, tune_size = 1000,
                                       kname = kern_param2n$kernelname, ktheta = kern_param2n$thetarel,
                                       kbetainv = kern_param2n$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))


# run gpr_sr
kern_param2 = readRDS('./kern_param2.rds')
bsize = 6750
nmodel = 500
update_k = 50
lr = 0.3
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 60 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> sr_bsize5000_nmodel500.txt &'))
t3=system.time(gbm_model3 <- gbm_train(ds2_trainmx, ds2_train_y, ds2_testmx, ds2_test_y, pred_method = "gbm_sr",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, tune_size = 1000,
                                       kname = "gaussiandotrel", ktheta = kern_param2$thetarel,
                                       kbetainv = kern_param2$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))


# run gbm4
kern_param2n = readRDS('./kern_param2n.rds')
bsize = 100
nmodel = 700
update_k = 50
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 60 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> gbm4_bsize100_nmodel700.txt &'))
t2=system.time(gbm_model2 <- gbm_train(ds2_trainmx, ds2_train_y, ds2_testmx, ds2_test_y, pred_method = "3",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = FALSE,
                                       update_kparam_tiems = update_k, tune_size = 1000, selected_n_feature = 45,
                                       kname = kern_param2n$kernelname, ktheta = kern_param2n$thetarel,
                                       kbetainv = kern_param2n$betainv))
system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))




# ols
lm.fit = lm(yy~.,data = ds2_train)
pred_ols = predict(lm.fit, newdata = as.data.frame(ds2_testmx))
rmse_ols <- rmse(pred_ols, ds2_test_y, "ols")

# xgboost
tmp_xgb <- xgboost(data = data.matrix(ds2_trainmx),
                   label = ds2_train_y,
                   eta = 0.05,
                   max_depth = 20,
                   nround = 200,
                   subsample = 0.3,
                   colsample_bytree = 1,
                   seed = 1
)
pred_xgb <- predict(tmp_xgb, data.matrix(ds2_testmx))
rmse_xgb <- rmse(pred_xgb, ds2_test_y, "xgboost")

# try kNN
pred_knn <- knn.reg(ds2_trainmx, ds2_testmx, ds2_train_y, k = 5)$pred
rmse_knn <- rmse(pred_knn, ds2_test_y, "knn")

# try LASSO
mdl_lasso = cv.glmnet(ds2_trainmx, ds2_train_y, family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = ds2_testmx)
rmse(pred_lasso, ds2_test_y, "LASSO")

# try RIDGE
lambdas <- 10^seq(1, -1, by = -0.01)
mdl_ridge = cv.glmnet(data.matrix(ds2_trainmx), data.matrix(ds2_train_y),
                      family = "gaussian", alpha = 0, lambda = lambdas)
pred_ridge = predict(mdl_ridge, newx = data.matrix(ds2_testmx))
rmse(pred_ridge, ds2_test_y, "RIDGE")

# try random forest
mdl_rf = ranger(yy ~ ., data = ds2_train, num.trees = 100, mtry = 12, write.forest = T)
pred_rf = predict(mdl_rf, ds2_test)
pred_rf2 = pred_rf$predictions
rmse(pred_rf2, ds2_test_y, "random forest")

# try svr ###########################
#  linear
time.linear = system.time(mdl_svr <- svm(yy~., ds2_train, kernel = "linear"))
svr.pred = predict(mdl_svr, ds2_testmx)
rmse_svr = rmse(svr.pred, ds2_test_y, "svr linear")

# poly
time.poly = system.time(mdl_svr <- svm(yy~., ds2_train, kernel = "polynomial", degree = 3))
svr.pred = predict(mdl_svr, ds2_testmx)
rmse_svr = rmse(svr.pred, ds2_test_y, "svr poly")

# rbf
time.rbf = system.time(mdl_svr <- tune.svm(yy~., data = ds2_train, kernel = "radial", gamma = 2^c(-8:0), cost = 2^c(-4:4)))
svr.pred = predict(mdl_svr$best.model, ds2_testmx)
rmse_svr = rmse(svr.pred, ds2_test_y, "svr rbf")





