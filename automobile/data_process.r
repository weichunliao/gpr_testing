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
# source('~/Desktop/baeirGPR/R/call_by_user.R')
# Rcpp::sourceCpp('~/Desktop/baeirGPR/src/matprod.cpp')
# source('~/Desktop/baeirGPR/R/gpr.R')
# source('~/Desktop/baeirGPR/R/local_gpr.R')
# source('~/Desktop/baeirGPR/R/gpr_tuning.R')
# source('~/Desktop/baeirGPR/R/boosting_gpr.R')
#####
#  -- Predicted price of car using all numeric and Boolean attributes
setwd("~/Desktop/gpr_testing/automobile")
ds <- fread("./imports-85.data")
missing_idx <- unique(which(ds == '?', arr.ind = T)[,1])

ds <- ds[-missing_idx,]
ds <- as.data.frame(ds)
attribute_names <- c("symboling", "normalized_losses", "make", "fuel_type", "aspiration",
                     "num_of_doors", "body_style", "drive_wheels", "engine_location", "wheel_base",
                     "length", "width", "height", "curb_weight", "engine_type",
                     "num_of_cylinders", "engine_size", "fuel_system", "bore", "stroke",
                     "compression_ratio", "horsepower", "peak_rpm", "city_mpg",
                     "highway_mpg", "price")

colnames(ds) <- attribute_names

ndata <- nrow(ds)
test_size <- floor(ndata/10)

set.seed(2018)
test_idx <- sample(ndata, test_size)

ds_train = ds[-test_idx,]
ds_train_x = ds_train[,-26]
ds_train_y = as.numeric(ds_train$price)

ds_test = ds[test_idx,]
ds_test_x = ds_test[,-26]
ds_test_y = as.numeric(ds_test$price)
# ########################
# for baeirgpr
dummy_names <- c("make", "fuel_type", "aspiration", "num_of_doors",
                 "body_style", "drive_wheels", "engine_location", "engine_type",
                 "num_of_cylinders", "fuel_system")
ds_dummy <- dummy.data.frame(ds, names = dummy_names)
dummy_cnames = names(ds_dummy)
dummy_cnames = gsub("-", "_", dummy_cnames)
ds_dummy <- as.data.frame(apply(ds_dummy, 2, as.numeric))
colnames(ds_dummy) <- dummy_cnames

# for (i in (1:ncol(ds_dummy))) {
#   cat(class(ds_dummy[,i]), "\n")
# }

ds2 = ds_dummy
#normalize to mean 0 and unit variance.
mean_shift = mean(ds_train_y)
ds2$price <- ds2$price - mean_shift

ds2_train = ds2[-test_idx,]
ds2_test = ds2[test_idx,]

for(ii in c(1,2,35,36,37,38,39,50,57,58,59,60,61,62,63)) {
  m1 <- mean(ds2_train[,ii])
  sd1 <- sd(ds2_train[,ii])
  ds2_train[,ii] <- (ds2_train[,ii] - m1)/sd1
  ds2_test[,ii] <- (ds2_test[,ii] - m1)/sd1
}

ds2_train_x <- model.matrix(price~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$price)
ds2_test_x <- model.matrix(price~.-1, ds2_test)
ds2_test_y <- as.matrix(ds2_test$price)

# baeirGPR
kern_param1 = gpr_tune(ds2_train_x, ds2_train_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
kern_param1 = gpr_tune(ds2_train_x, ds2_train_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50, init_betainv = kern_param1$betainv,
                       init_theta = kern_param1$thetarel)
saveRDS(kern_param1, "kern_param1.rds")
# nrow(ds2_train)
# ===================================================================
kern_param1 = readRDS('./kern_param1.rds')


bsize = 50
nmodel = 300
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> bsize_50_nmodel_300.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill -15 $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
t1
cat(" baeirGPR rmse =", tail(gbm_model1$test_rmse))

bsize = 50
nmodel = 300
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> gbm3_bsize_100_nmodel_200.txt &'))
t2=system.time(gbm_model2 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "3",
                                       n_model = nmodel, batch_size = bsize, lr = 0.001, tune_param = FALSE,
                                       update_kparam_tiems = update_k,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill -15 $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
t2
cat(" baeirGPR rmse =", tail(gbm_model2$test_rmse))

bsize = 110
nmodel = 300
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> sr_bsize_all_nmodel_300.txt &'))
t3=system.time(gbm_model3 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "gbm_sr",
                                       n_model = nmodel, batch_size = bsize, lr = 0.001, tune_param = TRUE,
                                       update_kparam_tiems = update_k, sr_size = 10,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill -15 $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
t3
cat(" baeirGPR rmse =", tail(gbm_model3$test_rmse))
###########################################

# ols
rm_col_idx = findLinearCombos(ds2_train_x)$remove
lm.fit = lm(price~.,data = ds2_train[,-rm_col_idx])
pred_ols = predict(lm.fit, newdata=as.data.frame(ds2_test_x[,-rm_col_idx]))
# lm.fit = lm(price~.,data = ds2_train)
# pred_ols = predict(lm.fit, newdata=ds2_test[,-64])
# pred_ols = pred_ols
rmse_xgb <- rmse(pred_ols, ds2_test_y, "ols")

# xgboost
tmp_xgb <- xgboost(data = data.matrix(ds2_train_x),
               label = ds2_train_y,
               eta = 0.05,
               max_depth = 15,
               nround = 100,
               subsample = 0.3,
               colsample_bytree = 1,
               seed = 1
)
pred_xgb <- predict(tmp_xgb, as.matrix(ds2_test_x))
rmse_xgb <- rmse(pred_xgb, ds2_test_y, "xgboost")

# full gpr
gpr_model1 <- traintraintrain(ds2_train_x, ds2_train_y, pred_method = "cg_direct_lm",
                              kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                              kbetainv = kern_param1$betainv, ncpu = -1, srsize = NULL,
                              clus_size = NULL)
gpr_pred1 <- gpr_fit(ds2_test_x, ds2_train_x, gpr_model1) + mean_shift
rmse_fullGPR <- rmse(gpr_pred1, ds_test_y, "full gpr")

# try kNN
pred_knn <- knn.reg(ds2_train_x, ds2_test_x, ds2_train_y, k = 3)$pred + mean_shift
rmse_knn <- rmse(pred_knn, ds_test_y, "knn")

# try LASSO
mdl_lasso = cv.glmnet(ds2_train_x, ds2_train_y, family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = ds2_test_x)
rmse(pred_lasso, ds2_test_y, "LASSO")

# try RIDGE
mdl_ridge = cv.glmnet(data.matrix(ds2_train_x), data.matrix(ds2_train_y),
                      family = "gaussian", alpha = 0)
pred_ridge = predict(mdl_ridge, newx = data.matrix(ds2_test_x))
rmse(pred_ridge, ds2_test_y, "RIDGE")

# try random forest
mdl_rf = ranger(price ~ ., data = ds2_train, num.trees = 50, mtry = 3, write.forest = T)
pred_rf = predict(mdl_rf, ds2_test_x)
pred_rf2 = pred_rf$predictions +mean_shift
rmse(pred_rf2, ds_test_y, "random forest")

# try svr
mdl_svr = svm(price~., ds2_train, kernel = "linear")
svr.pred = predict(mdl_svr, ds2_test_x)
rmse(svr.pred, ds2_test_y, "svr")


mdl_svr = tune.svm(price~., data = ds2_train, kernel = "radial", gamma = 2^c(-8:0), cost = 2^c(-2:6))
svr.pred = predict(mdl_svr$best.model, ds2_test_x)
rmse(svr.pred, ds2_test_y, "svr")
#
tuneResult <- tune(svm, price~.,  data = ds2_train,
                   ranges = list(epsilon = seq(0,1,0.01), cost = 2^(2:9))
)
tunedModel <- tuneResult$best.model
tunedModelY <- predict(tunedModel, ds2_test_x)
rmse(tunedModelY, ds2_test_y, "svr tune")


# =====================
#try vw
# vw_pred <- fread("./test.pred", header = F)
# vw_rmse <- sqrt(mean((ds_test_y- vw_pred)^2))
# cat(" vw rmse(w. online learning) = ", vw_rmse, "\n")
#
# vw_pred2 <- fread("./test2.pred", header = F)
# vw_rmse2 <- sqrt(mean((ds_test_y- vw_pred2)^2))
# cat(" vw rmse(without. online learning) = ", vw_rmse2, "\n")



#######################################

# for vw
# ds <- as.data.frame(ds)
# set.seed(2018)
# test_idx <- sample(ndata, test_size)
#
# ds_train_vw <- ds[-test_idx,]
# ds_test_vw <- ds[test_idx,]
#
# parse_line <- function(i_row) {
#   f_list <- paste(paste(names(i_row), i_row, sep  = ":"))
#   target_value <- paste(i_row[26], "|", collapse = " ")
#   feature_value <- paste(f_list[1:25], collapse = " ")
#   temp <- paste(target_value, feature_value, sep = " ", collapse = "")
#   return (temp)
# }
#
# tmp = ds_train_vw
# tmp_test = ds_test_vw
# output_form = apply(tmp, 1, function(x) parse_line(x))
# output_form_test = apply(tmp_test, 1, function(x) parse_line(x))
#
#
# fileConn<-file("vw_data.train")
# writeLines(output_form, fileConn)
# close(fileConn)
#
# fileConn<-file("vw_data.test")
# writeLines(output_form_test, fileConn)
# close(fileConn)
#
# fileConn<-file("vw_data2.test")
# writeLines(gsub(".*\\|", "|", output_form_test), fileConn)
# close(fileConn)
#
#
#
