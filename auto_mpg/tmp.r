library(data.table)

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
source('~/Desktop/baeirGPR/R/call_by_user.R')
Rcpp::sourceCpp('~/Desktop/baeirGPR/src/matprod.cpp')
source('~/Desktop/baeirGPR/R/gpr.R')
source('~/Desktop/baeirGPR/R/local_gpr.R')
source('~/Desktop/baeirGPR/R/gpr_tuning.R')
source('~/Desktop/baeirGPR/R/boosting_gpr.R')
#####
setwd("~/Desktop/gpr_testing/auto_mpg")
ds <- fread('./auto-mpg.csv', header = TRUE)
ds = ds[,-9]
f_names = c("mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration",
            "model_year", "origin")

missing_idx <- unique(which(ds == "?", arr.ind = T)[,1])

ds <- ds[-missing_idx,]
ds <- as.data.frame(apply(ds,2,as.numeric))
names(ds) = f_names

ndata <- nrow(ds)
test_size <- floor(ndata/10)

set.seed(2018)
test_idx <- sample(ndata, test_size)
# ds = apply(ds, 2, as.numeric)
# mean shift
ds$mpg <- ds$mpg - mean(ds$mpg)
# ds = apply(ds,2,as.numeric)
# train-test split
ds_train = ds[-test_idx,]
ds_train_x = ds_train[,-1]
ds_train_y = ds_train$mpg

ds_test = ds[test_idx,]
ds_test_x = ds_test[,-1]
ds_test_y = ds_test$mpg

# try baeirGPR
ds2_train_x = ds_train_x
ds2_train_y = ds_train_y
ds2_test_x = ds_test_x
ds2_test_y = ds_test_y

for(ii in 2:(ncol(ds2_train_x))) {
  m1 <- mean(ds2_train_x[,ii])
  sd1 <- sd(ds2_train_x[,ii])

  ds2_train_x[,ii] = (ds2_train_x[,ii] - m1)/sd1
  ds2_test_x[,ii] = (ds2_test_x[,ii] - m1)/sd1
}

ds2_train_x <- model.matrix(~.-1, ds2_train_x)
ds2_train_y <- as.matrix(ds2_train_y)

ds2_test_x <- model.matrix(~.-1, ds2_test_x)
ds2_test_y <- as.matrix(ds2_test_y)

# kern_param1 = gpr_tune(ds2_train_x, ds2_train_y, kernelname = "rbf",
#                        optim_report = 5, optim_ard_report = 5,
#                        ARD = T, optim_ard_max = 50)
# kern_param1 = gpr_tune(ds2_train_x, ds2_train_y, kernelname = "rbf",
#                        optim_report = 5, optim_ard_report = 5,
#                        ARD = T, optim_ard_max = 50, init_betainv = kern_param1$betainv,
#                        init_theta = kern_param1$thetarel)
# saveRDS(kern_param1, file='kern_param1.rds')
# ===============================
kern_param1 = readRDS('kern_param1.rds')
bsize = 50
nmodel = 300
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 5 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> bsize_50_nmodel_200.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill -15 $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
#

bsize = 50
nmodel = 300
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> gbm3_bsize_50_nmodel_300.txt &'))
t2=system.time(gbm_model2 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "3",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = FALSE,
                                       update_kparam_tiems = update_k,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill -15 $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))

#
bsize = 110
nmodel = 300
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> sr_bsize_110_nmodel_300.txt &'))
t3=system.time(gbm_model3 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "gbm_sr",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, sr_size = 10,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill -15 $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))


cat(" baeirGPR rmse =", tail(gbm_model1$test_rmse))
cat(" baeirGPR rmse =", tail(gbm_model2$test_rmse))
cat(" baeirGPR rmse =", tail(gbm_model3$test_rmse))

# try ols
lm.fit = lm(mpg~.,data = ds_train)
pred_ols = predict(lm.fit, newdata=ds_test)
rmse_ols <- rmse(pred_ols, ds_test_y, "ols")

# xgboost
n_feature = ncol(ds_train_x)
tmp_xgb <- xgboost(data = data.matrix(ds_train_x),
                   label = ds_train_y,
                   eta = 0.025,
                   max_depth = 6,
                   nround = 750,
                   subsample = 0.7,
                   nthread = 4,
                   colsample_bytree = 1,
                   seed = 1
)
pred_xgb <- predict(tmp_xgb, data.matrix(ds_test_x))
rmse_xgb <- rmse(pred_xgb, ds_test_y, "xgboost")

# full gpr
gpr_model1 <- traintraintrain(ds2_train_x, ds2_train_y, pred_method = "cg_direct_lm",
                              kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                              kbetainv = kern_param1$betainv, ncpu = -1, srsize = NULL,
                              clus_size = NULL)
gpr_pred1 <- gpr_fit(ds2_test_x, ds2_train_x, gpr_model1)
rmse_fullGPR <- rmse(gpr_pred1, ds_test_y, "full gpr")

# try kNN
pred_knn <- knn.reg(ds_train_x, ds_test_x, ds_train_y, k = 3)$pred
rmse_knn <- rmse(pred_knn, ds_test_y, "knn")

# try LASSO
mdl_lasso = cv.glmnet(data.matrix(ds_train_x), data.matrix(ds_train_y), family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = data.matrix(ds_test_x))
rmse(pred_lasso, ds_test_y, "LASSO")

# try RIDGE
lambdas <- 10^seq(30, -40, by = -.01)
mdl_ridge = cv.glmnet(data.matrix(ds_train_x), data.matrix(ds_train_y),
                      family = "gaussian", alpha = 0, lambda = lambdas)
pred_ridge = predict(mdl_ridge, newx = data.matrix(ds_test_x))
rmse(pred_ridge, ds_test_y, "RIDGE")

# try random forest
mdl_rf = ranger(mpg ~ ., data = ds_train, num.trees = 50, mtry = 3, write.forest = T)
pred_rf = predict(mdl_rf, ds_test_x)
rmse(pred_rf$predictions, ds_test_y, "random forest")

# try vw
vw_pred <- fread("./test.pred", header = F)
vw_rmse <- sqrt(mean((ds_test_y- vw_pred)^2))
cat(" vw rmse(w. online learning) = ", vw_rmse, "\n")

vw_pred2 <- fread("./test2.pred", header = F)
vw_rmse2 <- sqrt(mean((ds_test_y- vw_pred2)^2))
cat(" vw rmse(without. online learning) = ", vw_rmse2, "\n")



# ################# make file for vw ###############
ds_train_vw <- ds[-test_idx,]
ds_test_vw <- ds[test_idx,]

parse_line <- function(i_row) {
  f_list <- paste(paste(names(i_row), i_row, sep  = ":"))
  target_value <- paste(i_row[1], "|", collapse = " ")
  feature_value <- paste(f_list[2:8], collapse = " ")
  temp <- paste(target_value, feature_value, sep = " ", collapse = "")
  return (temp)
}
# tmp = head(ds)
# tmp_test = tail(ds)

tmp = ds_train_vw
tmp_test = ds_test_vw
output_form = apply(tmp, 1, function(x) parse_line(x))
output_form_test = apply(tmp_test, 1, function(x) parse_line(x))

fileConn<-file("vw_data.train")
writeLines(output_form, fileConn)
close(fileConn)

fileConn<-file("vw_data.test")
writeLines(output_form_test, fileConn)
close(fileConn)

fileConn<-file("vw_data2.test")
writeLines(gsub(".*\\|", "|", output_form_test), fileConn)
close(fileConn)
