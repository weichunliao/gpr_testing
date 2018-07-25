library(data.table)
library(dummies)

# library(ranger)

# library(forecast)

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


setwd('~/Desktop/gpr_testing/sandp/')
tmp = readRDS('./data5.rds')

tmp$log_return = NULL
tmp = as.data.frame(tmp)

dummy_names <- c("year", "month", "weekday")
ds_dummy <- dummy.data.frame(tmp, names = dummy_names)
dummy_cnames = names(ds_dummy)
ds_dummy <- as.data.frame(apply(ds_dummy, 2, as.numeric))
colnames(ds_dummy) <- dummy_cnames

ds_train = ds_dummy
ds_train_y = as.matrix(ds_train$Volume_adj)

ds2 = ds_dummy
ds2$instant <- log(c(1:nrow(ds_train)))
# mean_shift = mean(ds_train$Volume_adj)
# ds2$Volume_adj = ds2$Volume_adj - mean_shift
ds2$Volume_adj = log2(ds2$Volume_adj)
mean_shift = mean(log2(ds_train$Volume_adj))
ds2$Volume_adj = ds2$Volume_adj - mean_shift

ds2_train = ds2

ds2_train_x <- model.matrix(Volume_adj~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$Volume_adj)

########## tune kern_param ##########
n_data = nrow(ds2_train)
tune_ind <- sample(n_data, 1000)
kern_param1 = gpr_tune(ds2_train_x[1:1000,], ds2_train_y[1:1000,], kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
# tune_ind <- sample(n_data, 2000)
kern_param1 = gpr_tune(ds2_train_x[tune_ind,], ds2_train_y[tune_ind,], kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50, init_betainv = kern_param1$betainv,
                       init_theta = kern_param1$thetarel)
saveRDS(kern_param1, './kern_param1_volume.rds')
###################################

# function param
train_x = ds2_train_x
train_y = ds2_train_y
lr = 0.05
train_window_size <- 1000
move_window_size <- 30

######

kparam = readRDS('./kern_param1_volume.rds')
n_data = length(train_y)
tune_size = 1000

adj_all_y <- train_y

# pred_all_y <- rep(0, n_data)
final_pred_all_y_gbm <- rep(0, length(train_y))
#
n_model = ceiling((n_data - train_window_size)/move_window_size)

train_start_idx <- 1
train_end_idx <- train_start_idx + train_window_size - 1
#
record_pred_start_idx <- train_end_idx + 1
record_pred_end_idx <- record_pred_start_idx + move_window_size - 1

sub_rmse_list = rep(0, n_model)
rmse_list = rep(0, n_model)
cat('n_model: ', n_model, '\n')
t0 = Sys.time()
# n_model = 5
for (iter_id in c(1:n_model)) {
  cat("Now, running for iteration", iter_id, "\n")
  if (record_pred_end_idx > n_data) {
    record_pred_end_idx <- n_data
  }

  if(iter_id %% 10 == 0) {
    cat('  update kparam ...\n')
    tune_ind <- c((train_end_idx-tune_size):train_end_idx)
    kparam <- gpr_tune(train_x[tune_ind,], adj_all_y[tune_ind],
                       ARD = T, init_betainv = kparam$betainv,
                       init_theta = kparam$thetarel)
    # kparam <- gpr_tune(train_x[tune_ind,], adj_all_y[tune_ind],
    #                    ARD = F, init_betainv = kparam$betainv,
    #                    init_theta = kparam$thetarel)
  }

  cat(train_start_idx, train_end_idx, record_pred_start_idx, record_pred_end_idx, '\n')
  train_ind <- c(train_start_idx:train_end_idx)

  for (j in c(5)){
    new_iter_id <- (iter_id-1)*5 +j
    cat('new_iter_id', new_iter_id, '\n')
    temp_model <- gpr_train(train_x[train_ind,], adj_all_y[train_ind], kparam)

    tmp_pred_all_y <- gpr_predict(train_x[(record_pred_start_idx:n_data),], train_x[train_ind,], temp_model)

    if (new_iter_id > 1) {
      adj_all_y[record_pred_start_idx:n_data] <- adj_all_y[record_pred_start_idx:n_data] - tmp_pred_all_y * lr
    } else {
      adj_all_y[record_pred_start_idx:n_data] <- adj_all_y[record_pred_start_idx:n_data] - tmp_pred_all_y
    }

    if (new_iter_id > 1) {
      final_pred_all_y_gbm[record_pred_start_idx:n_data] = final_pred_all_y_gbm[record_pred_start_idx:n_data] +
        tmp_pred_all_y * lr
    } else {
      final_pred_all_y_gbm[record_pred_start_idx:n_data] = final_pred_all_y_gbm[record_pred_start_idx:n_data] +
        tmp_pred_all_y
    }
  }

  rmse_list[iter_id] = rmse(train_y[record_pred_start_idx:n_data],final_pred_all_y_gbm[record_pred_start_idx:n_data])
  sub_rmse_list[iter_id] = rmse(final_pred_all_y_gbm[record_pred_start_idx:record_pred_end_idx],train_y[record_pred_start_idx:record_pred_end_idx])


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

rmse(ds_train_y[(train_window_size+1):n_data], 2^(final_pred_all_y_gbm[(train_window_size+1):n_data]+mean_shift))

rmse(ds2_train_y[(train_window_size+1):n_data], (final_pred_all_y_gbm[(train_window_size+1):n_data]))


plot(tail(ds_train_y[1:1100], 100), type='l', col='black', ylim = c(1,5000), ylab='prediction', xlab = 'period')
lines(tail(2^(final_pred_all_y_gbm+mean_shift)[1:1100], 100), col='red')
lines(tail(2^(final_pred_all_y_ar+mean_shift)[1:1100], 100), col='green')
legend('topright', horiz = T, fill=c('black', 'red'), legend=c('orginal data', 'GBGPR'))


plot(tail(ds2_train_y, 100), type='l', col='black', ylim=c(-3,15))
lines(tail((final_pred_all_y+mean_shift), 100), col='red')

