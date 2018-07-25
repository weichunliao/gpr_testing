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
setwd('~/Desktop/gpr_testing/bike/')
tmp = fread('./hour.csv')
tmp = as.data.frame(tmp)

feature_names <- c("instant", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
                   "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt")
tmp2 = tmp[,feature_names]

dummy_names <- c("season", "yr", "mnth", "hr",  "weekday", "weathersit")
ds_dummy <- dummy.data.frame(tmp2, names = dummy_names)
dummy_cnames = names(ds_dummy)
ds_dummy <- as.data.frame(apply(ds_dummy, 2, as.numeric))
colnames(ds_dummy) <- dummy_cnames

ndata = nrow(ds_dummy)
ds = cbind(ds_dummy[2:ndata,1:56], ds_dummy[1:(ndata-1), 57:62], ds_dummy[2:ndata,63])
colnames(ds) = dummy_cnames

ds_train = ds
ds_train_y = as.matrix(ds_train$cnt)

ds2 = ds
ds2[,1] <- log(ds2[,1]) ### log(t_period)
# mean_shift = mean(ds_train$cnt)
# ds2$cnt = ds2$cnt - mean_shift
ds2$cnt = log2(ds2$cnt)
mean_shift = mean(log2(ds_train$cnt))
ds2$cnt = ds2$cnt - mean_shift

ds2_train = ds2

for (i in c(57:62)){
  m1 <- mean(ds2_train[,i])
  sd1 <- sd(ds2_train[,i])
  ds2_train[,i] <- (ds2_train[,i] - m1)/sd1
}

ds2_train_x <- model.matrix(cnt~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$cnt)

######################################
n_data = nrow(ds2_train)
# tune_ind <- sample(n_data, 1000)
# kern_param1 = gpr_tune(ds2_train_x[tune_ind,], ds2_train_y[tune_ind,], kernelname = "rbf",
#                        optim_report = 5, optim_ard_report = 5,
#                        ARD = T, optim_ard_max = 50)
# tune_ind <- sample(n_data, 2000)
# kern_param1 = gpr_tune(ds2_train_x[tune_ind,], ds2_train_y[tune_ind,], kernelname = "rbf",
#                        optim_report = 5, optim_ard_report = 5,
#                        ARD = T, optim_ard_max = 50, init_betainv = kern_param1$betainv,
#                        init_theta = kern_param1$thetarel)
# saveRDS(kern_param1, './kern_param3_time_log.rds')
#####
# tune_ind <- sample(n_data, 2000)
# kern_param1n = gpr_tune(ds2_train_x[tune_ind,], ds2_train_y[tune_ind,], kernelname = "rbf",
#                        optim_report = 5, optim_ard_report = 5,
#                        ARD = F, optim_ard_max = 50)
# tune_ind <- sample(n_data, 2000)
# kern_param1n = gpr_tune(ds2_train_x[tune_ind,], ds2_train_y[tune_ind,], kernelname = "rbf",
#                        optim_report = 5, optim_ard_report = 5,
#                        ARD = F, optim_ard_max = 50, init_betainv = kern_param1n$betainv,
#                        init_theta = kern_param1n$thetarel)
# saveRDS(kern_param1n, './kern_param3n_time_log.rds')

####
# function param
train_x = ds2_train_x
train_y = ds2_train_y
lr = 0.005
train_window_size <- 1440
move_window_size <- 168

######
# kparam = readRDS('./kern_param1_time_log.rds')
kparam = readRDS('./kern_param3_time_log.rds')
# kparam = readRDS('./kern_param3n_time_log.rds')
# kparam = kern_param1n
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
cat('n_model: ', n_model, '\n')
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
    # kparam <- gpr_tune(train_x[tune_ind,], adj_all_y[tune_ind],
    #                    ARD = F, init_betainv = kparam$betainv,
    #                    init_theta = kparam$thetarel)
  }

  cat(train_start_idx, train_end_idx, record_pred_start_idx, record_pred_end_idx, '\n')
  train_ind <- c(train_start_idx:train_end_idx)

  for (j in c(2)){
    new_iter_id <- (iter_id-1)*2 +j
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

##ttt
plot(tail(2^(final_pred_all_y_arima[(train_window_size+1):n_data]+mean_shift),250), type='l', col='blue',ylim = c(0,500), xlab='period', ylab='prediction')
# plot(tail(ds_train_y[(train_window_size+1):n_data],250), type='l', col='blue',ylim = c(0,500), xlab='period', ylab='prediction')

lines(tail(2^(final_pred_all_y_ar[(train_window_size+1):n_data]+mean_shift),250), col='green')
lines(tail(ds_train_y[(train_window_size+1):n_data],250), col='black')
lines(tail(2^(final_pred_all_y[(train_window_size+1):n_data]+mean_shift),250), col='red')

par(xpd=TRUE)
legend(30,650, legend=c('original data', 'GBGPR', 'ar', 'arima'),
       fill = c('black', 'red', 'green', 'blue'), horiz = T)
saveRDS(final_pred_all_y, 'gbgbm_168.rds')


plot(ds_train_y[1:1440], type='l')


rmse(ds2_train_y[(train_window_size+1):n_data], (final_pred_all_y[(train_window_size+1):n_data]))
rmse(ds_train_y[(train_window_size+1):n_data], 2^(final_pred_all_y_ar[(train_window_size+1):n_data]+mean_shift))
