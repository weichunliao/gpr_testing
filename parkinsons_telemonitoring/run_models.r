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
source('~/Desktop/baeirGPR/R/call_by_user.R')
Rcpp::sourceCpp('~/Desktop/baeirGPR/src/matprod.cpp')
source('~/Desktop/baeirGPR/R/gpr.R')
source('~/Desktop/baeirGPR/R/local_gpr.R')
source('~/Desktop/baeirGPR/R/gpr_tuning.R')
source('~/Desktop/baeirGPR/R/boosting_gpr.R')
#####
setwd("~/Desktop/gpr_testing/parkinsons_telemonitoring")

# The main aim of the data is to predict the
# motor and total UPDRS scores ('motor_UPDRS' and 'total_UPDRS') from the 16
# voice measures.
ds <- fread('~/Desktop/gpr_testing/parkinsons_telemonitoring/parkinsons_updrs.data')

ds <- as.data.frame(apply(ds, 2, as.numeric))
feature_names <- c('subjuct_num', 'age', 'sex', 'test_time', 'motor_UPDRS',
              'total_UPDRS', 'jitter_percent', 'jitter_abs', 'jitter_rap',
              'jitter_ppq5', 'jitter_ddp', 'shimmer', 'shimmer_db', 'shimmer_apq3',
              'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr',
              'rpde', 'dfa', 'ppe')
setnames(ds, feature_names)


ndata <- nrow(ds)
test_size <- floor(ndata/10)

set.seed(2018)
test_idx <- sample(ndata, test_size)

ds_train <- ds[-test_idx,]
y_col_idx = which(names(ds_train) %in% c("motor_UPDRS", "total_UPDRS") , arr.ind = T)
ds_train_x <- model.matrix(~.-1, ds_train[,-c(y_col_idx)])
ds_train_y1 <- as.matrix(ds_train$motor_UPDRS)
ds_train_y2 <- as.matrix(ds_train$total_UPDRS)

ds_test <- ds[test_idx,]
ds_test_x <- model.matrix(~.-1, ds_test[,-c(y_col_idx)])
ds_test_y1 <- as.matrix(ds_test$motor_UPDRS)
ds_test_y2 <- as.matrix(ds_test$total_UPDRS)

#
ds2_x <- ds[,-c(y_col_idx)]
ds2_train_x <- ds_train[,-c(y_col_idx)]
# mean shift
ds2_y1 <- ds$motor_UPDRS - mean(ds_train_y1)
ds2_y2 <- ds$total_UPDRS - mean(ds_train_y2)
# normalize features
for (ii in 1:ncol(ds2_x) ) {
  m_i <- mean(ds_train_x[,ii])
  std_i <- sd(ds_train_x[,ii])
  ds2_x[,ii] <- (ds2_x[,ii] - m_i)/std_i
}

###
ds2_train_x <- model.matrix(~.-1, ds2_x[-test_idx,])
ds2_train_y1 <- as.matrix(ds2_y1[-test_idx])
ds2_train_y2 <- as.matrix(ds2_y2[-test_idx])
ds2_test_x <- model.matrix(~.-1, ds2_x[test_idx,])
ds2_test_y1 <- as.matrix(ds2_y1[test_idx])
ds2_test_y2 <- as.matrix(ds2_y2[test_idx])

# try baeirGPR
t_size <- 1000
t_idx <- sample(nrow(ds_train), t_size)
ds_tune_x <- ds2_train_x[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y1[t_idx,])
# ds_tune_y <- as.matrix(ds2_train_y2[t_idx,])
kern_param1 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
kern_param1 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50, init_betainv = kern_param1$betainv,
                       init_theta = kern_param1$thetarel)
saveRDS(kern_param1, "kern_param1.rds")
############
kern_param1n = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = F, optim_ard_max = 50)
kern_param1n = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = F, optim_ard_max = 50, init_betainv = kern_param1n$betainv,
                       init_theta = kern_param1$thetarel)
saveRDS(kern_param1n, "kern_param1n.rds")


######################################################################################################


kern_param1 = readRDS("kern_param1.rds")
ds2_train_y <- ds2_train_y1
# ds2_train_y <- ds2_train_y2
ds2_test_y <- ds2_test_y1
# ds2_test_y <- ds2_test_y2



# gpr2
bsize = 50
nmodel = 300
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 15 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> gbm2_bsize_all_nmodel300.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))






#gpr3
bsize = 50
nmodel = 300
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 15 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> gbm2_bsize_all_nmodel300.txt &'))
t2=system.time(gbm_model2 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "3",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = FALSE,
                                       update_kparam_tiems = update_k,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))

#gpr sr
bsize = 50
nmodel = 300
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 15 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> gbm2_bsize_all_nmodel300.txt &'))
t3=system.time(gbm_model3 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "gbm_sr",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
cat(" baeirGPR rmse =", tail(gbm_model1$test_rmse))
cat(" baeirGPR rmse (col sampling) =", tail(gbm_model2$test_rmse))
cat(" baeirGPR rmse (gbm sr)=", tail(gbm_model3$test_rmse))

# try ols
lm.fit <- lm(motor_UPDRS~.,data = ds_train[,-6])
pred_ols <- predict(lm.fit, newdata=ds_test[,-6])
rmse_ols <- rmse(pred_ols, ds_test_y1, "ols")

# try xgboost
n_feature = ncol(ds_train_x)
tmp_xgb <- xgboost(data = data.matrix(ds_train_x),
                   label = ds_train_y1,
                   eta = 0.025,
                   max_depth = 6,
                   nround = 750,
                   subsample = 0.7,
                   nthread = 4,
                   colsample_bytree = 1,
                   seed = 1
)
pred_xgb <- predict(tmp_xgb, ds_test_x)
rmse_xgb <- rmse(pred_xgb, ds_test_y1, "xgboost")

# full gpr
gpr_model1 <- traintraintrain(ds2_train_x, ds2_train_y, pred_method = "cg_direct_lm",
                              kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                              kbetainv = kern_param1$betainv, ncpu = -1, srsize = NULL,
                              clus_size = NULL)
gpr_pred1 <- gpr_fit(ds2_test_x, ds2_train_x, gpr_model1)
rmse_fullGPR <- rmse(gpr_pred1, ds2_test_y1, "full gpr")

# try kNN
pred_knn1 <- knn.reg(ds_train_x, ds_test_x, ds_train_y1, k = 8)$pred
rmse_knn <- rmse(pred_knn1, ds_test_y1, "knn")

# try LASSO
mdl_lasso = cv.glmnet(ds_train_x, ds_train_y1, family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = ds_test_x)
rmse(pred_lasso, ds_test_y1, "LASSO")

# try RIDGE
mdl_ridge = cv.glmnet(ds_train_x, ds_train_y1, family = "gaussian", alpha = 0)
pred_ridge = predict(mdl_ridge, newx = ds_test_x)
rmse(pred_ridge, ds_test_y1, "RIDGE")

# try random forest
f = 'motor_UPDRS ~ .'
mdl_rf = ranger(f, ds_train[,-6], num.trees = 10, mtry = 5, write.forest = T)
pred_rf = predict(mdl_rf, ds_test_x)
rmse_rf = rmse(pred_rf$predictions, ds_test_y1, "random forest")

# ################# make file for vw ###############
ds_train_vw_y1 <- ds_train[,-6]
ds_train_vw_y2 <- ds_train[,-5]
ds_test_vw_y1 <- ds_test[,-6]
ds_test_vw_y2 <- ds_test[,-5]

parse_line <- function(i_row) {
  f_list <- paste(paste(names(i_row), i_row, sep  = ":"))
  target_value <- paste(i_row[5], "|", collapse = " ")
  feature_value <- paste(f_list[-5], collapse = " ")
  temp <- paste(target_value, feature_value, sep = " ", collapse = "")
  return (temp)
}

tmp_y1 = ds_train_vw_y1
tmp_y2 = ds_train_vw_y2
tmp_test_y1 = ds_test_vw_y1
tmp_test_y2 = ds_test_vw_y2

output_form_y1 = apply(tmp_y1, 1, function(x) parse_line(x))
output_form_y2 = apply(tmp_y2, 1, function(x) parse_line(x))
output_form_test_y1 = apply(tmp_test_y1, 1, function(x) parse_line(x))
output_form_test_y2 = apply(tmp_test_y2, 1, function(x) parse_line(x))











