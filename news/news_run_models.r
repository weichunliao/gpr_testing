library(data.table)

library(xgboost)
library(FNN)
library(glmnet)
library(ranger)

library(baeirGPR)
#####
rmse <- function(y_hat, y, method = "") {
  out <- sqrt(mean((y - y_hat)^2))
  cat(method,"rmse = ", out, "\n")
  return(out)
}
#####
setwd('~/Desktop/gpr_testing/news/')

tmp = fread('./OnlineNewsPopularity.csv')
tmp$url = NULL

# ds = as.data.frame(tmp[(tmp$shares<5000),])
ds = as.data.frame(tmp)

set.seed(2018)
n_data = nrow(ds)
test_size = floor(n_data/10)
test_id = sample(n_data, test_size)

ds_train = ds[-test_id,]
ds_test = ds[test_id,]
ds_test_y = ds_test$shares

# hist(log(ds_train$shares))
# hist(log(ds_test$shares))

# hist(ds_train$shares)

# ds = as.data.frame(tmp[(tmp$shares<5000),])
#
# hist(ds$shares)
# hist(log2(ds$shares))
# nrow(ds)

ds2 = as.data.frame(ds)
# shift
mean_shift = mean(log2(ds_train$shares))
ds2$shares = log2(ds2$shares) - mean_shift

for (i in c(1:12,19:30)) {
  # i = 1
  m1 = mean(ds_train[,i])
  s1 = sd(ds_train[,i])
  ds2[,i] = (ds2[,i]-m1)/s1
  # i= i+1
}

ds2_train = ds2[-test_id,]
ds2_train_x = model.matrix(shares~.-1, ds2_train)
ds2_train_y = as.matrix(ds2_train$shares)
ds2_test = ds2[test_id,]
ds2_test_x = model.matrix(shares~.-1, ds2_test)
ds2_test_y = as.matrix(ds2_test$shares)



######################
t_size <- 2000
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
t_size <- 2000
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

############## run gbm ############
# gpr2
kern_param1 = readRDS('kern_param1.rds')
bsize = 3000
nmodel = 7
update_k = 50
lr = 0.05
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 15 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> gbm2_bsize_5000.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, tune_size = 1000,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))

# run gbm3 ##################################
kern_param1n = readRDS('./kern_param1n.rds')
bsize =100
nmodel = 700
update_k = 50
lr = 0.05
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 30 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> gbm3_bsize500.txt &'))
t2=system.time(gbm_model2 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "3",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = FALSE,
                                       update_kparam_tiems = update_k, tune_size = 2000,
                                       kname = kern_param1n$kernelname, ktheta = kern_param1n$thetarel,
                                       kbetainv = kern_param1n$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))


# run gbm_sr ##################################
kern_param1 = readRDS('./kern_param1.rds')
bsize = 3000
nmodel = 5
update_k = 50
lr = 0.1
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 30 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> sr_bsize5000.txt &'))
t3=system.time(gbm_model3 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "gbm_sr",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, tune_size = 2000, decay_lr = 0.9,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))




#############
# ========================================= #
# try ols
lm.fit <- lm(shares~.,data = ds2_train)
pred_ols <- predict(lm.fit, newdata=ds2_test)
## no log
rmse_ols <- rmse(pred_ols, ds2_test_y, "ols")
## log
pred_ols = pred_ols+mean_shift
pred_ols = 2^(pred_ols)
rmse_ols <- rmse(pred_ols, ds_test_y, "ols")


# try xgb
n_feature = ncol(ds2_train_x)
tmp_xgb <- xgboost(data = data.matrix(ds2_train_x),
                   label = ds2_train_y,
                   eta = 0.05,
                   max_depth = 6,
                   nround = 300,
                   subsample = 0.7,
                   nthread = 4,
                   colsample_bytree = 1,
                   seed = 1
)
pred_xgb <- predict(tmp_xgb, ds2_test_x)
# pred_xgb = pred_xgb + mean_shift
# pred_xgb = 2^pred_xgb
rmse_xgb <- rmse(pred_xgb, ds2_test_y, "xgboost")



