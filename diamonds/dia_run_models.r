library(data.table)
library(dummies)

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
setwd('~/Desktop/gpr_testing/diamonds/')

tmp = fread('./diamonds.csv')
tmp$V1 = NULL
tmp = as.data.frame(tmp)

dummy_names <- c("cut", "color", "clarity")
ds_dummy <- dummy.data.frame(tmp, names = dummy_names)
dummy_cnames = names(ds_dummy)
ds_dummy <- as.data.frame(apply(ds_dummy, 2, as.numeric))
colnames(ds_dummy) <- dummy_cnames

ds = ds_dummy[,c("carat", "cutFair", "cutGood", "cutIdeal", "cutPremium", "cutVery Good", "colorD", "colorE", "colorF", "colorG",
                  "colorH", "colorI", "colorJ", "clarityI1", "clarityIF", "claritySI1", "claritySI2", "clarityVS1", "clarityVS2", "clarityVVS1",
                  "clarityVVS2", "depth",  "table",  "x",  "y", "z", "price")]

ndata <- nrow(ds)
test_size <- floor(ndata/10)

set.seed(2018)
test_idx <- sample(ndata, test_size)

ds_train = ds[-test_idx,]
ds_test = ds[test_idx,]
ds_test_y = ds_test$price

ds2 = as.data.frame(ds)
# normalize to mean 0 and unit variance.
mean_shift = mean(log(ds_train$price))
ds2$price <- log(ds2$price) - mean_shift

# mean_shift = mean(ds_train$price)
# ds2$price = ds2$price - mean_shift

# feature normalize
for (i in c(1,22:26)){
  m1 = mean(ds_train[,i])
  sd1 = sd(ds_train[,i])
  ds2[,i] = (ds2[,i] - m1)/sd1
}

ds2_train = ds2[-test_idx,]
ds2_test = ds2[test_idx,]
ds2_train_x <- model.matrix(price~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$price)
ds2_test_x <- model.matrix(price~.-1, ds2_test)
ds2_test_y <- as.matrix(ds2_test$price)


# tune
t_size <- 1000
t_idx <- sample(nrow(ds2_train_x), t_size)
ds_tune_x <- ds2_train_x[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param2 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
t_idx <- sample(nrow(ds2_train_x), t_size)
ds_tune_x <- ds2_trainmx[t_idx,]
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
ds_tune_x <- ds2_trainmx[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
kern_param1n = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                        optim_report = 5, optim_ard_report = 5,
                        ARD = F, optim_ard_max = 50, init_betainv = kern_param1n$betainv,
                        init_theta = kern_param1n$thetarel)
saveRDS(kern_param1n, "kern_param1n.rds")

# ===================================================================
kern_param1 = readRDS("kern_param1.rds")

# gpr2
bsize = 100
nmodel = 500
update_k = 20
lr = 0.01
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 15 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> gbm2_bsize_100_nmodel500.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, tune_size = 1000,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))


#gpr3
bsize = 100
nmodel = 700
update_k = 20
lr = 0.01
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 15 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> gbm3_bsize100_nmodel700.txt &'))
t2=system.time(gbm_model2 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "3",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = FALSE,
                                       update_kparam_tiems = update_k, tune_size = 1000,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))


# gpr sr
bsize = 670
nmodel = 500
update_k = 20
lr = 0.01
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 15 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> sr_bsize670_nmodel500.txt &'))
t3=system.time(gbm_model3 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "gbm_sr",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, tune_size = 1000,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))

# ========================================= #
# try ols
lm.fit <- lm(price~.,data = ds2_train)
pred_ols <- predict(lm.fit, newdata=ds2_test)
pred_ols = exp(pred_ols+mean_shift)
# rmse_ols <- rmse(pred_ols, ds2_test_y, "ols")
rmse_ols <- rmse(pred_ols, ds_test_y, "ols")

# try xgboost
n_feature = ncol(ds2_train_x)
tmp_xgb <- xgboost(data = data.matrix(ds2_train_x),
                   label = ds2_train_y,
                   eta = 0.1,
                   max_depth = 6,
                   nround = 100,
                   subsample = 0.7,
                   nthread = 4,
                   colsample_bytree = 1,
                   seed = 1
)
pred_xgb <- predict(tmp_xgb, ds2_test_x)
rmse_xgb <- rmse(pred_xgb, ds2_test_y, "xgboost")


# try kNN
pred_knn <- knn.reg(ds2_train_x, ds2_test_x, ds2_train_y, k = 8)$pred
rmse_knn <- rmse(pred_knn1, ds2_test_y, "knn")

# try LASSO
mdl_lasso = cv.glmnet(ds2_train_x, ds2_train_y, family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = ds2_test_x)
rmse(pred_lasso, ds2_test_y, "LASSO")

# try RIDGE
lambdas <- 10^seq(1, -1, by = -0.0001)
mdl_ridge = cv.glmnet(data.matrix(ds2_train_x), data.matrix(ds2_train_y),
                      family = "gaussian", alpha = 0, lambda = lambdas)
pred_ridge = predict(mdl_ridge, newx = data.matrix(ds2_test_x))
rmse(pred_ridge, ds2_test_y, "RIDGE")

# try random forest
mdl_rf = ranger(price ~ ., data = ds2_train, num.trees = 50, mtry = 3, write.forest = T)
pred_rf = predict(mdl_rf, ds2_test_x)
pred_rf2 = pred_rf$predictions +mean_shift
rmse(pred_rf2, ds_test_y, "random forest")

# try svr ###########################
# linear
time.linear = system.time(mdl_svr <- svm(price~., ds2_train, kernel = "linear"))
svr.pred = predict(mdl_svr, ds2_test_x)
rmse_svr = rmse(svr.pred, ds2_test_y, "svr linear")

# poly
time.poly = system.time(mdl_svr <- svm(price~., ds2_train, kernel = "polynomial", degree = 3))
svr.pred = predict(mdl_svr, ds2_test_x)
rmse_svr = rmse(svr.pred, ds2_test_y, "svr poly")

# rbf
time.rbf = system.time(mdl_svr <- tune.svm(price~., data = ds2_train, kernel = "radial", gamma = 2^c(-8:0), cost = 2^c(-4:4)))
svr.pred = predict(mdl_svr$best.model, ds2_test_x)
rmse_svr = rmse(svr.pred, ds2_test_y, "svr rbf")
