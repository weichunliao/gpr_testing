library(data.table)
library(dummies)

library(ranger)

library(baeirGPR)
library(forecast)
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
# log period
ds2$instant = log(ds2$instant)
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
saveRDS(kern_param1, "kern_param_28_log.rds")
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
kern_param1 = readRDS('./kern_param_28_log.rds')
bsize = 500
nmodel = 10
update_k = 50
lr = 0.05
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 30 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> gbm2_bsize3000.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, decay_lr = 0.9, tune_size = 2000,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
n_th_list = c(1,10,50,100,200,300,400,500,600,700,800,900,1000)
n_th_list = c(1,10)
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
bsize = 3000
nmodel = 10
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
# n_th_list = c(1,10,50,100,200,300,400,500)
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


# n_th_list = c(1,10,50,100,200,300,400,500,600,700,800,900,1000)
# o_model = gbm_model2
# rmse_list = vector("numeric")
# for (n in n_th_list) {
#   n_th = n
#   test_pred = gbm_fit(ds2_test_x, ds2_test_y, ds2_train_x,
#                       list(models = o_model$models[c(1:n_th)], pred_method=o_model$pred_method))$prediction
#   test_pred = test_pred + mean_shift
#   test_pred = 2^test_pred
#   rmse_list = c(rmse_list, rmse(test_pred, ds_test_y))
# }
# for (i in rmse_list){cat(i, '\n')}
#
#
#
#
# test_pred = gbm_fit(ds2_test_x, ds2_test_y, ds2_train_x,
#                     list(models = gbm_model3$models[1], pred_method=gbm_model3$pred_method))
o_model = readRDS('./gbm3_005_3000.rds')
rmse_list = vector("numeric")
n_th = 10
test_pred = gbm_fit(ds2_test_x, ds2_test_y, ds2_train_x,
                    list(models = o_model$models[c(1:n_th)], pred_method=o_model$pred_method))$prediction
test_pred = test_pred + mean_shift
test_pred = 2^test_pred
rmse_list = c(rmse_list, rmse(test_pred, ds_test_y))
plot(ds_test_y[1:240], type='l', col='blue')
lines(test_pred[1:240], type='l', col='red')
for (i in rmse_list){cat(i, '\n')}

gbm_model4= readRDS('./gbm3_005_100.rds')
rmse_list = vector("numeric")
n = 100
n_th = n
test_pred2 = gbm_fit(ds2_test_x, ds2_test_y, ds2_train_x,
                    list(models = gbm_model4$models[c(1:n_th)], pred_method=gbm_model4$pred_method))$prediction
test_pred2 = test_pred2 + mean_shift
test_pred2 = 2^test_pred2
rmse_list = c(rmse_list, rmse(test_pred2, ds_test_y))
lines(test_pred2[1:240], type='l', col='green')

ccc = c("blue", "red", "purple", "black")
ddd = c("true y", "gpr", "ar", "arima")
legend("topright", legend = ddd, col=, fill=ccc)
############
# ar
cnt_train = ds$cnt[1:train_idx]
cnt_test = ds$cnt[(train_idx+1):nrow(ds)]

aaa = ar(cnt_train, order.max = 26)
p = predict(aaa, n.ahead=length(cnt_test))$pred
lines(p[1:240], type='l', col='purple')
rmse(p, cnt_test)
# arima
# zz= auto.arima(cnt_train)


a2 <- auto.arima(ts(cnt_train,frequency = 24),D=1)
tsp2<-forecast(a2, h=length(cnt_test))
# tsp2<-forecast(a2, h=240)
lines(as.numeric(tsp2$mean[1:240]), col='black')
# plot(as.numeric(tsp2$mean[1:240]), col='black', type="l")
##########################################################
# plot(tail(ds_test_y,240), type='l', col='blue')
# lines(tail(test_pred,240), type='l', col='red')




a2 <- arima(cnt_train, order=c(12,0,0))
# tsp2<-forecast(a2, h=length(cnt_test))
tsp2<-forecast(a2, h=240)
plot(as.numeric(tsp2$mean[1:240]), col='black', type='l')
a3 = ar(cnt_train, order.max = 12)
tsp3 = predict(a3, n.ahead=length(cnt_test))$pred
lines(tsp3[1:240], type='l', col='red')

######################################################################################################################
lm.fit = lm(cnt~.,data = ds2_train)
pred_ols = predict(lm.fit, newdata = as.data.frame(ds2_test_x))
pred_ols = 2^(pred_ols+mean_shift)
rmse_ols = rmse(pred_ols, ds_test_y)
# rmse_ols <- rmse(pred_ols, ds2_test_y, "ols")

# xgboost
library(xgboost)

tmp_xgb <- xgboost(data = data.matrix(ds2_train_x),
                   label = ds2_train_y,
                   eta = 0.1,
                   max_depth = 20,
                   nround = 300,
                   subsample = 0.7,
                   colsample_bytree = 1,
                   seed = 1
)
pred_xgb <- predict(tmp_xgb, data.matrix(ds2_test_x))
pred_xgb = 2^(pred_xgb+mean_shift)
rmse_xgb <- rmse(pred_xgb, ds_test_y, "xgboost")

# try kNN
library(FNN)
pred_knn <- knn.reg(ds2_train_x, ds2_test_x, ds2_train_y, k = 5)$pred
pred_knn = 2^(pred_knn+mean_shift)
rmse_knn <- rmse(pred_knn, ds_test_y, "knn")

# try LASSO
library(glmnet)
mdl_lasso = cv.glmnet(ds2_train_x, ds2_train_y, family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = ds2_test_x)
pred_lasso=2(pred_lasso+mean_shift)
rmse(pred_lasso, ds_test_y, "LASSO")

# try RIDGE
lambdas <- 10^seq(5, -5, by = -0.01)
mdl_ridge = cv.glmnet(data.matrix(ds2_train_x), data.matrix(ds2_train_y),
                      family = "gaussian", alpha = 0, lambda = lambdas)
pred_ridge = predict(mdl_ridge, newx = data.matrix(ds2_test_x))
pred_ridge = 2^(pred_ridge+mean_shift)
rmse(pred_ridge, ds_test_y, "RIDGE")

# try random forest
mdl_rf = ranger(cnt ~ ., data = ds2_train, num.trees = 100, mtry = 13, write.forest = T)
pred_rf = predict(mdl_rf, ds2_test)
pred_rf2 = pred_rf$predictions
pred_rf2 = 2^(pred_rf2+mean_shift)
rmse(pred_rf2, ds_test_y, "random forest")

# try svr ###########################
library(e1071)
#  linear
time.linear = system.time(mdl_svr <- svm(cnt~., ds2_train, kernel = "linear"))
svr.pred = predict(mdl_svr, ds2_test_x)
svr.pred =2^(svr.pred+mean_shift)
rmse_svr = rmse(svr.pred, ds_test_y, "svr linear")

# poly
time.poly = system.time(mdl_svr <- svm(cnt~., ds2_train, kernel = "polynomial", degree = 3))
svr.pred = predict(mdl_svr, ds_test_x)
svr.pred =2^(svr.pred+mean_shift)
rmse_svr = rmse(svr.pred, ds_test_y, "svr poly")

# rbf
time.rbf = system.time(mdl_svr <- tune.svm(cnt~., data = ds2_train[1:2000,], kernel = "radial", gamma = 2^c(-8:0), cost = 2^c(-4:4)))
svr.pred = predict(mdl_svr$best.model, ds2_test_x)
svr.pred =2^(svr.pred+mean_shift)
rmse_svr = rmse(svr.pred, ds_test_y, "svr rbf")












