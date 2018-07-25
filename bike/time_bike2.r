library(data.table)
library(dummies)

library(ranger)

library(forecast)

# library(baeirGPR)
######
source('~/Desktop/baeirGPR/R/call_by_user.R')
Rcpp::sourceCpp('~/Desktop/baeirGPR/src/matprod.cpp')
source('~/Desktop/baeirGPR/R/gpr.R')
source('~/Desktop/baeirGPR/R/local_gpr.R')
source('~/Desktop/baeirGPR/R/gpr_tuning.R')
source('~/Desktop/baeirGPR/R/boosting_gpr.R')
###########
rmse <- function(y_hat, y, method = "") {
  out <- sqrt(mean((y - y_hat)^2))
  cat(method,"rmse = ", out, "\n")
  return(out)
}
#####
setwd('~/Desktop/gpr_testing/bike/')
tmp = fread('./hour.csv')
tmp = as.data.frame(tmp)

feature_names <- c("instant", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
                   "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt")
tmp2 = tmp[,feature_names]

dummy_names <- c("season", "weekday", "weathersit")
ds_dummy <- dummy.data.frame(tmp2, names = dummy_names)
dummy_cnames = names(ds_dummy)
ds_dummy <- as.data.frame(apply(ds_dummy, 2, as.numeric))
colnames(ds_dummy) <- dummy_cnames

ndata = nrow(ds_dummy)
ds = cbind(ds_dummy[2:ndata,1:17], ds_dummy[1:(ndata-1), 18:27], ds_dummy[2:ndata,28])
colnames(ds) = dummy_cnames

# test_size <- floor(ndata/10)
# train_idx <- ndata-test_size-1
train_idx <- ndata
ds_train = ds
ds_train_y <- as.matrix(ds_train$cnt)
# ds_test = ds[(train_idx+1):nrow(ds),]
# ds_test_y <- as.matrix(ds_test$cnt)

ds2 = ds
# log period
ds2$instant = log(ds2$instant)
# normalize cnt
ds2$cnt = log2(ds2$cnt)
mean_shift = mean(log2(ds_train$cnt))
ds2$cnt = ds2$cnt - mean_shift

ds2_train = ds2
# ds2_test = ds2[(train_idx+1):nrow(ds),]

for (i in c(22:27)){
  m1 <- mean(ds2_train[,i])
  sd1 <- sd(ds2_train[,i])
  ds2_train[,i] <- (ds2_train[,i] - m1)/sd1
  # ds2_test[,i] <- (ds2_test[,i] - m1)/sd1
}

ds2_train_x <- model.matrix(cnt~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$cnt)
# ds2_test_x <- model.matrix(cnt~.-1, ds2_test)
# ds2_test_y <- as.matrix(ds2_test$cnt)

######################################


######################################
# n_data = nrow(ds2_train)
# tune_ind <- sample(n_data, 2000)
# kern_param1 = gpr_tune(ds2_train_x[tune_ind,], ds2_train_y[tune_ind,], kernelname = "rbf",
#                        optim_report = 5, optim_ard_report = 5,
#                        ARD = T, optim_ard_max = 50)
# tune_ind <- sample(n_data, 2000)
# kern_param1 = gpr_tune(ds2_train_x[tune_ind,], ds2_train_y[tune_ind,], kernelname = "rbf",
#                        optim_report = 5, optim_ard_report = 5,
#                        ARD = T, optim_ard_max = 50, init_betainv = kern_param1$betainv,
#                        init_theta = kern_param1$thetarel)
# saveRDS(kern_param1, './.rds')
####
# function param
train_x = ds2_train_x
train_y = ds2_train_y
lr = 0.01
train_window_size <- 1680
move_window_size <- 168

######
kparam = readRDS('./kern_param_28_log.rds')
# kparam = kern_param1
n_data = length(train_y)
tune_size = 1000

adj_all_y <- train_y

# pred_all_y <- rep(0, n_data)
final_pred_all_y <- rep(0, length(train_y))
#
n_model = ceiling((n_data - train_window_size)/move_window_size)

train_start_idx <- 1
train_end_idx <- train_start_idx + train_window_size - 1
#
record_pred_start_idx <- train_end_idx + 1
record_pred_end_idx <- record_pred_start_idx + move_window_size - 1

sub_rmse_list = rep(0, n_model)
rmse_list = rep(0, n_model)

t0 = Sys.time()
for (iter_id in c(1:n_model)) {
  cat("Now, running for iteration", iter_id, "\n")
  if (record_pred_end_idx > n_data) {
    record_pred_end_idx <- n_data
  }

  if(iter_id %% 10 == 0) {
    cat('  update kparam ...\n')
    tune_ind <- sample(train_end_idx, tune_size)
    kparam <- gpr_tune(train_x[tune_ind,], adj_all_y[tune_ind],
                       ARD = T, init_betainv = kparam$betainv,
                       init_theta = kparam$thetarel)
  }

  # cat(train_start_idx, train_end_idx, record_pred_start_idx, record_pred_end_idx, '\n')
  train_ind <- c(train_start_idx:train_end_idx)

  # temp_model <- gpr_train(train_x[train_ind,], adj_all_y[train_ind], kparam)
  #
  # tmp_pred_all_y <- gpr_predict(train_x[(record_pred_start_idx:n_data),], train_x[train_ind,], temp_model)
  #
  # if (iter_id > 1) {
  #   adj_all_y[record_pred_start_idx:n_data] <- adj_all_y[record_pred_start_idx:n_data] - tmp_pred_all_y * lr
  # } else {
  #   adj_all_y[record_pred_start_idx:n_data] <- adj_all_y[record_pred_start_idx:n_data] - tmp_pred_all_y
  # }
  #
  # if (iter_id > 1) {
  #   final_pred_all_y[record_pred_start_idx:n_data] = final_pred_all_y[record_pred_start_idx:n_data] +
  #     tmp_pred_all_y * lr
  # } else {
  #   final_pred_all_y[record_pred_start_idx:n_data] = final_pred_all_y[record_pred_start_idx:n_data] +
  #     tmp_pred_all_y
  # }
  for (j in c(1:3)){
    new_iter_id <- (iter_id-1)*3 +j
    cat('new_iter_id', new_iter_id, '\n')
    temp_model <- gpr_train(train_x[train_ind,], adj_all_y[train_ind], kparam)

    tmp_pred_all_y <- gpr_predict(train_x[(record_pred_start_idx:n_data),], train_x[train_ind,], temp_model)

    if (new_iter_id > 1) {
      adj_all_y[record_pred_start_idx:n_data] <- adj_all_y[record_pred_start_idx:n_data] - tmp_pred_all_y * lr
    } else {
      adj_all_y[record_pred_start_idx:n_data] <- adj_all_y[record_pred_start_idx:n_data] - tmp_pred_all_y
    }

    if (new_iter_id > 1) {
      final_pred_all_y[record_pred_start_idx:n_data] = final_pred_all_y[record_pred_start_idx:n_data] +
        tmp_pred_all_y * lr
    } else {
      final_pred_all_y[record_pred_start_idx:n_data] = final_pred_all_y[record_pred_start_idx:n_data] +
        tmp_pred_all_y
    }
  }


  rmse_list[iter_id] = rmse(train_y[record_pred_start_idx:n_data],final_pred_all_y[record_pred_start_idx:n_data])
  sub_rmse_list[iter_id] = rmse(final_pred_all_y[record_pred_start_idx:record_pred_end_idx],train_y[record_pred_start_idx:record_pred_end_idx])


  train_start_idx <- train_start_idx + move_window_size
  train_end_idx <- train_end_idx + move_window_size
  record_pred_start_idx <- record_pred_start_idx + move_window_size
  record_pred_end_idx <- record_pred_end_idx + move_window_size
}

t1 = Sys.time()
rmse_list
sub_rmse_list
print(t1-t0)

###############################

rmse(ds_train_y[(train_window_size+1):n_data], 2^(final_pred_all_y[(train_window_size+1):n_data]+mean_shift))

rmse(ds2_train_y[(train_window_size+1):n_data], (final_pred_all_y[(train_window_size+1):n_data]))


plot(tail(ds_train_y[(train_window_size+1):n_data],240), type='l', col='blue')
lines(tail(2^(final_pred_all_y[(train_window_size+1):n_data]+mean_shift),240), col='red')

rmse(ds_train_y[(train_window_size+1):n_data], 2^(final_pred_all_y[(train_window_size+1):n_data]+mean_shift))









###### 測試區 ############
kern_param1n = readRDS('./kern_param1n.rds')

#good
gbm_model2 = readRDS('./gbm3_005_3000.rds')
gbm_model3 = readRDS('./gbm3_005_100.rds')
gbm_model4 = readRDS('./gbm3_01_4000.rds')

#bad
gbm_model02 = readRDS('./gbm3_005_2000.rds')
gbm_model03 = readRDS('./gbm3_01_5000.rds')

########################################################
kparam = readRDS('./kern_param1n.rds')
# used_col = gbm_model2$models[[1]]$col_sampling_idx
# used_col = c(4,5,6,8,9,12,14,16,17,18,21,23,27)
used_col = c(2:21)
ds2_train_x2 = ds2_train_x[,used_col]
ds2_test_x2 = ds2_test_x[,used_col]

n_data = nrow(ds2_train)
ds2_train_x[,1] = log(ds2_train_x[,1])
tune_ind <- sample(n_data, 1000)
kern_param1 = gpr_tune(ds2_train_x[tune_ind,], ds2_train_y[tune_ind,], kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
kparam = kern_param1
#
kparam <- gpr_tune(ds2_train_x2[tune_ind,], ds2_train_y[tune_ind,],
                   ARD = F, init_betainv = kparam$betainv,
                   init_theta = kparam$thetarel)
####
train_ind <- c(1:240)
# train_ind <- sample(2400, 1000)
temp_model <- gpr_train(ds2_train_x2[train_ind,], ds2_train_y[train_ind], kparam)

pred_test_y <- gpr_predict(ds2_train_x2[c(241:264),], ds2_train_x2[train_ind,], temp_model)
rmse_test <- sqrt(mean((ds2_train_y[c(241:264)] - pred_test_y)^2))
rmse_test

########
train_ind <- sample(n_data, 1000)
# train_ind <- c(1:1000)
temp_model <- gpr_train(ds2_train_x2[train_ind,], ds2_train_y[train_ind], kparam)

pred_test_y <- gpr_predict(ds2_test_x2, ds2_train_x2[train_ind,], temp_model)
rmse_test <- sqrt(mean((ds2_test_y - pred_test_y)^2))
rmse_test
# > gbm_model2$test_rmse[1]
# [1] 0.5189475


############

ds2_train_x[,1] = log(ds2_train_x[,1])
train_ind <- c(1:2400)
# train_ind <- sample(n_data, 3000)
temp_model <- gpr_train(ds2_train_x[train_ind,], ds2_train_y[train_ind], kern_param1)

pred_test_y <- gpr_predict(ds2_train_x[2401:1400,], ds2_train_x[train_ind,], temp_model)
rmse_test <- sqrt(mean((ds2_train_y[2401:1400] - pred_test_y)^2))
rmse_test

