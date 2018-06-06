library(data.table)

library(xgboost)
library(FNN)
library(glmnet)
library(ranger)

# library(baeirGPR)
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
setwd("~/Desktop/gpr_testing/concrete")

# Name -- Data Type -- Measurement -- Description
# Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
# Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
# Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
# Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
# Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
# Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
# Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
# Age -- quantitative -- Day (1~365) -- Input Variable
# Concrete compressive strength -- quantitative -- MPa -- Output Variable
ds <- fread('./concrete_data.csv', header = T, encoding = 'UTF-8')
ds = as.data.frame(ds)

ndata <- nrow(ds)
test_size <- floor(ndata/10)

set.seed(2018)
test_idx <- sample(ndata, test_size)

ds_train = ds[-test_idx,]
ds_train_x = ds_train[, c(1:(ncol(ds_train)-1))]
ds_train_y = as.matrix(ds_train$concrete_compressive_strength)

ds_test = ds[test_idx,]
ds_test_x = ds_test[, c(1:(ncol(ds_train)-1))]
ds_test_y = as.matrix(ds_test$concrete_compressive_strength)


####### for baeirGPR #####################
ds2 <- as.data.frame(ds)

ds2$concrete_compressive_strength = as.numeric(ds2$concrete_compressive_strength)

ds2$concrete_compressive_strength <- ds2$concrete_compressive_strength - mean(ds_train$concrete_compressive_strength)
mean_shift = mean(ds_train$concrete_compressive_strength)

ds2_train = ds2[-test_idx,]
ds2_test = ds2[test_idx,]

for(ii in 1:(ncol(ds2)-1)) {
  m1 <- mean(ds2_train[,ii])
  sd1 <- sd(ds2_train[,ii])
  ds2_train[,ii] <- (ds2_train[,ii] - m1)/sd1
  ds2_test[,ii] <- (ds2_test[,ii] - m1)/sd1
}

ds2_train_x <- model.matrix(concrete_compressive_strength~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$concrete_compressive_strength)

ds2_test_x <- model.matrix(concrete_compressive_strength~.-1, ds2_test)
ds2_test_y <- as.matrix(ds2_test$concrete_compressive_strength)

ncol(ds2_train)
nrow(ds2_train)
# try baeirGPR
t_size <- 900
t_idx <- sample(nrow(ds_train), t_size)
ds_tune_x <- ds2_train_x[t_idx,]
ds_tune_y <- as.matrix(ds2_train_y[t_idx,])
# kern_param1 = gpr_tune(ds2_train_x, ds2_train_y, kernelname = "rbf",
#                        optim_report = 5, optim_ard_report = 5,
#                        ARD = T, optim_ard_max = 50)
kern_param1 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
kern_param1 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50, init_betainv = kern_param1$betainv,
                       init_theta = kern_param1$thetarel)
saveRDS(kern_param1, 'kern_param1.rds')
# =================
kern_param1 = readRDS('kern_param1.rds')
bsize = 100
nmodel = 600
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 15 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> gbm2_bsize_100_nmodel600.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
t1
cat(" baeirGPR rmse =", tail(gbm_model1$test_rmse))
cat("Min baeirGPR (row)  rmse =", min(gbm_model1$test_rmse))
# gpr col
kern_param1 = readRDS('kern_param1.rds')
bsize = 100
nmodel = 600
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 15 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> gbm3_bsize_100_nmodel600.txt &'))
t2=system.time(gbm_model2 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "3",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = FALSE,
                                       update_kparam_tiems = update_k,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))

cat(" baeirGPR rmse =", tail(gbm_model2$test_rmse))
cat("Min baeirGPR (col) rmse =", min(gbm_model2$test_rmse))
# gpr sr
bsize = 670
nmodel = 10
update_k = 6
lr = 0.01
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> sr_bsize_670_nmodel300.txt &'))
t3=system.time(gbm_model3 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "gbm_sr",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
t3
cat(" baeirGPR rmse =", tail(gbm_model3$test_rmse))
cat("Min baeirGPR (sr) rmse =", min(gbm_model3$test_rmse))


# try ols
lm.fit <- lm(concrete_compressive_strength~.,data = ds_train)
pred_ols <- predict(lm.fit, newdata=ds_test)
rmse_ols <- rmse(pred_ols, ds_test_y, "ols")

# try xgboost
n_feature = ncol(ds_train_x)
tmp_xgb <- xgboost(data = data.matrix(ds_train_x),
                   label = ds_train_y,
                   eta = 0.025,
                   max_depth = 6,
                   nround = 150,
                   subsample = 0.7,
                   nthread = 4,
                   colsample_bytree = 1,
                   seed = 1
)
pred_xgb <- predict(tmp_xgb, as.matrix(ds_test_x))
rmse_xgb <- rmse(pred_xgb, ds_test_y, "xgboost")

# full gpr
gpr_model1 <- traintraintrain(ds2_train_x, ds2_train_y, pred_method = "cg_direct_lm",
              kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
              kbetainv = kern_param1$betainv, ncpu = -1, srsize = NULL,
              clus_size = NULL)
gpr_pred1 <- gpr_fit(ds2_test_x, ds2_train_x, gpr_model1) + mean_shift
rmse_fullGPR <- rmse(gpr_pred1, ds_test_y, "full gpr")

# try kNN
pred_knn <- knn.reg(ds_train_x, ds_test_x, ds_train_y, k = 3)$pred
rmse_knn <- rmse(pred_knn, ds_test_y, "knn")

# try LASSO
mdl_lasso = cv.glmnet(ds_train_x, ds_train_y, family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = ds_test_x)
rmse(pred_lasso, ds_test_y, "LASSO")

# try RIDGE
mdl_ridge = cv.glmnet(ds_train_x, ds_train_y, family = "gaussian", alpha = 0)
pred_ridge = predict(mdl_ridge, newx = ds_test_x)
rmse(pred_ridge, ds_test_y, "RIDGE")

# try random forest
mdl_rf = ranger(concrete_compressive_strength ~ ., data = ds_train, num.trees = 10, mtry = 5, write.forest = T)
pred_rf = predict(mdl_rf, ds_test_x)
rmse_rf = rmse(pred_rf$predictions, ds_test_y, "random forest")

# ################# make file for vw ###############
ds_train_vw <- ds[-test_idx,]
ds_test_vw <- ds[test_idx]

parse_line <- function(i_row) {
  f_list <- paste(paste(names(i_row), i_row, sep  = ":"))
  target_value <- paste(i_row[9], "|", collapse = " ")
  feature_value <- paste(f_list[1:8], collapse = " ")
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



