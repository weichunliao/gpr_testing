library(data.table)
library(dummies)

library(ranger)

library(baeirGPR)
######
rmse <- function(y_hat, y, method = "") {
  out <- sqrt(mean((y - y_hat)^2))
  cat(method,"rmse = ", out, "\n")
  return(out)
}
#####
setwd('~/Desktop/gpr_testing/bike/')
tmp = fread('./hour.csv')
tmp = as.data.frame(tmp)

# for (i in c(1:ncol(tmp))) {
#   cat(length(unique(tmp[,i])), '\n')
# }

# n = 48
# plot(tmp$instant[1:n], tmp$cnt[1:n], type = "l")
# head(tmp,50)

# for (i in c(1:ncol(tmp))) {
#   print(class(tmp[,i]))
# }


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

test_size <- floor(ndata/10)
train_idx <- ndata-test_size-1
ds_train = ds[1:train_idx,]
ds_test = ds[(train_idx+1):nrow(ds),]
ds_test_y <- as.matrix(ds_test$cnt)

ds2 = ds
ds2$cnt = log2(ds2$cnt)
mean_shift = mean(log2(ds_train$cnt))
ds2$cnt = ds2$cnt - mean_shift

ds2_train = ds2[1:train_idx,]
ds2_test = ds2[(train_idx+1):nrow(ds),]

for (i in c(22:27)){
  m1 <- mean(ds2_train[,i])
  sd1 <- sd(ds2_train[,i])
  ds2_train[,i] <- (ds2_train[,i] - m1)/sd1
  ds2_test[,i] <- (ds2_test[,i] - m1)/sd1
}

ds2_train_x <- model.matrix(cnt~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$cnt)
ds2_test_x <- model.matrix(cnt~.-1, ds2_test)
ds2_test_y <- as.matrix(ds2_test$cnt)

# tune kernel param
t_size <- 2000
t_idx <- sample(nrow(ds2_train_x), t_size)
ds_tune_x <- ds2_train_x[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param1 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
# t_idx <- sample(nrow(ds2_train_x), t_size)
# ds_tune_x <- ds2_train_x[t_idx,]
# ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
# kern_param1 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
#                        optim_report = 5, optim_ard_report = 5,
#                        ARD = T, optim_ard_max = 50, init_betainv = kern_param1$betainv,
#                        init_theta = kern_param1$thetarel)
saveRDS(kern_param1, "kern_param1.rds")
#### no ard
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
##########

#run gbm2
kern_param1 = readRDS('./kern_param1.rds')
bsize = 500
nmodel = 1000
update_k = 50
lr = 0.1
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 30 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> gbm2_bsize3000.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, decay_lr = 0.9, tune_size = 2000,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
n_th_list = c(1,10,50,100,200,300,400,500,600,700,800,900,1000)
# n_th_list = c(1,10)
o_model = gbm_model1
rmse_list = vector("numeric")
for (n in n_th_list) {
  n_th = n
  test_pred = gbm_fit(ds2_test_x, ds2_test_y, ds2_train_x,
                      list(models = o_model$models[c(1:n_th)], pred_method=o_model$pred_method))$prediction
  test_pred = test_pred + mean_shift
  test_pred = 2^test_pred
  rmse_list = c(rmse_list, rmse(test_pred, ds_test_y))
}
for (i in rmse_list){cat(i, '\n')}

# run gbm3 ##################################
kern_param1n = readRDS('./kern_param1n.rds')
bsize = 1000
nmodel = 1000
update_k = 50
lr = 0.05
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 30 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> gbm3_bsize1000.txt &'))
t2=system.time(gbm_model2 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "3",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = FALSE,
                                       update_kparam_tiems = update_k, tune_size = 2000,
                                       kname = kern_param1n$kernelname, ktheta = kern_param1n$thetarel,
                                       kbetainv = kern_param1n$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
n_th_list = c(1,10,50,100,200,300,400,500,600,700,800,900,1000)
o_model = gbm_model2
rmse_list = vector("numeric")
for (n in n_th_list) {
  n_th = n
  test_pred = gbm_fit(ds2_test_x, ds2_test_y, ds2_train_x,
                      list(models = o_model$models[c(1:n_th)], pred_method=o_model$pred_method))$prediction
  test_pred = test_pred + mean_shift
  test_pred = 2^test_pred
  rmse_list = c(rmse_list, rmse(test_pred, ds_test_y))
}
for (i in rmse_list){cat(i, '\n')}


# run gbm_sr ##################################
kern_param1 = readRDS('./kern_param1.rds')
bsize = 100
nmodel = 10
update_k = 50
lr = 0.1
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 30 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> sr_bsize8950.txt &'))
t3=system.time(gbm_model3 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "gbm_sr",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, tune_size = 2000, decay_lr = 0.9,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
# n_th_list = c(1,10,50,100,200,300,400,500,600,700,800,900,1000)
n_th_list = c(1,10)
o_model = gbm_model3
rmse_list = vector("numeric")
for (n in n_th_list) {
  n_th = n
  test_pred = gbm_fit(ds2_test_x, ds2_test_y, ds2_train_x,
                      list(models = o_model$models[c(1:n_th)], pred_method=o_model$pred_method))$prediction
  test_pred = test_pred + mean_shift
  test_pred = 2^test_pred
  rmse_list = c(rmse_list, rmse(test_pred, ds_test_y))
}
for (i in rmse_list){cat(i, '\n')}


n_th_list = c(1,10,50,100,200,300,400,500,600,700,800,900,1000)
o_model = gbm_model2
rmse_list = vector("numeric")
for (n in n_th_list) {
  n_th = n
  test_pred = gbm_fit(ds2_test_x, ds2_test_y, ds2_train_x,
                      list(models = o_model$models[c(1:n_th)], pred_method=o_model$pred_method))$prediction
  test_pred = test_pred + mean_shift
  test_pred = 2^test_pred
  rmse_list = c(rmse_list, rmse(test_pred, ds_test_y))
}
for (i in rmse_list){cat(i, '\n')}




test_pred = gbm_fit(ds2_test_x, ds2_test_y, ds2_train_x,
                    list(models = gbm_model3$models[1], pred_method=gbm_model3$pred_method))






