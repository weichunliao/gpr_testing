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

##########
train_x = ds2_train_x
train_y = ds2_train_y
lr = 0.01
train_window_size <- 1000
move_window_size <- 30

#
n_data = length(train_y)
n_model = ceiling((n_data - train_window_size)/move_window_size)

final_pred_all_y_ar <- rep(0, length(train_y))
train_start_idx <- 1
train_end_idx <- train_start_idx + train_window_size - 1
#
record_pred_start_idx <- train_end_idx + 1
record_pred_end_idx <- record_pred_start_idx + move_window_size - 1


for (iter_id in c(1:n_model)) {
  if (record_pred_end_idx > n_data) {
    record_pred_end_idx <- n_data
  }
  cat(train_start_idx, train_end_idx, record_pred_start_idx, record_pred_end_idx, '\n')

  cnt_train = train_y[1:train_end_idx]
  cnt_test = train_y[record_pred_start_idx:record_pred_end_idx]

  ar_model1 = ar(cnt_train)
  p = predict(ar_model1, n.ahead=length(cnt_test))$pred
  # plot(p, type='l', col='purple')
  # plot(cnt_test)
  # rmse(p, cnt_test)

  final_pred_all_y_ar[record_pred_start_idx:record_pred_end_idx] = p

  train_start_idx <- train_start_idx + move_window_size
  train_end_idx <- train_end_idx + move_window_size
  record_pred_start_idx <- record_pred_start_idx + move_window_size
  record_pred_end_idx <- record_pred_end_idx + move_window_size
}

rmse(final_pred_all_y_ar[(train_window_size+1):n_data], train_y[(train_window_size+1):n_data])

rmse(2^final_pred_all_y_ar[(train_window_size+1):n_data], 2^(train_y[(train_window_size+1):n_data]+mean_shift))

plot(train_y[1680:2000], col='black', type='l')
lines(final_pred_all_y_ar[1680:2000],col='red')
saveRDS(final_pred_all_y_ar, 'ar_168.rds')
#################################################
################################### arima ######################


train_x = ds2_train_x
train_y = ds2_train_y
lr = 0.01
train_window_size <- 1000
move_window_size <- 30

#
n_data = length(train_y)
n_model = ceiling((n_data - train_window_size)/move_window_size)

final_pred_all_y_arima <- rep(0, length(train_y))
train_start_idx <- 1
train_end_idx <- train_start_idx + train_window_size - 1
#
record_pred_start_idx <- train_end_idx + 1
record_pred_end_idx <- record_pred_start_idx + move_window_size - 1

# n_model=2
for (iter_id in c(1:n_model)) {
  if (record_pred_end_idx > n_data) {
    record_pred_end_idx <- n_data
  }
  cat(train_start_idx, train_end_idx, record_pred_start_idx, record_pred_end_idx, '\n')

  cnt_train = train_y[1:train_end_idx]
  cnt_test = train_y[record_pred_start_idx:record_pred_end_idx]

  a2 <- arima(cnt_train, order = c(3L,1L,1L))
  tsp2<-forecast(a2, h=length(cnt_test))
  # a2 <- arima(cnt_train, order=c(12,0,0))
  # tsp2<-forecast(a2, h=length(cnt_test))
  # tsp2<-forecast(a2, h=240)
  # plot(as.numeric(tsp2$mean), col='black', type='l', ylim=c(-3,0))
  # lines(cnt_test, col='red')

  # rmse(tsp2$mean, cnt_test)

  final_pred_all_y_arima[record_pred_start_idx:record_pred_end_idx] = tsp2$mean

  train_start_idx <- train_start_idx + move_window_size
  train_end_idx <- train_end_idx + move_window_size
  record_pred_start_idx <- record_pred_start_idx + move_window_size
  record_pred_end_idx <- record_pred_end_idx + move_window_size
}

rmse(final_pred_all_y_arima[(train_window_size+1):n_data], train_y[(train_window_size+1):n_data])

rmse(2^final_pred_all_y_arima[(train_window_size+1):n_data], 2^(train_y[(train_window_size+1):n_data]+mean_shift))


plot(ds_train_y[1001:n_data], type='l', col='black', ylab='trade volume', xlab='period')
lines(2^(final_pred_all_y_ar+mean_shift)[1001:n_data], col='green')
lines(2^(final_pred_all_y_arima+mean_shift)[1001:n_data], col='blue')
lines(2^(final_pred_all_y_gbm+mean_shift)[1001:n_data], col='red')

par(xpd=TRUE)
legend(30,14000, legend=c('original data', 'GBGPR', 'ar', 'arima'),
       fill = c('black', 'red', 'green', 'blue'), horiz = T)

rmse(final_pred_all_y_ar[(train_window_size+1):n_data], train_y[(train_window_size+1):n_data])
rmse(final_pred_all_y_arima[(train_window_size+1):n_data], train_y[(train_window_size+1):n_data])
rmse(final_pred_all_y_gbm[(train_window_size+1):n_data], train_y[(train_window_size+1):n_data])


rmse(2^(final_pred_all_y_ar[(train_window_size+1):n_data]+mean_shift), ds_train_y[(train_window_size+1):n_data])
rmse(2^(final_pred_all_y_arima[(train_window_size+1):n_data]+mean_shift), ds_train_y[(train_window_size+1):n_data])
rmse(2^(final_pred_all_y_gbm[(train_window_size+1):n_data]+mean_shift), ds_train_y[(train_window_size+1):n_data])


