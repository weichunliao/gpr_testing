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
setwd("~/Desktop/gpr_testing/concrete_slump")
# Input variables (7)(component kg in one M^3 concrete):
#   Cement
#   Slag
#   Fly ash
#   Water
#   SP
#   Coarse Aggr.
#   Fine Aggr.
#
# Output variables (3):
#   SLUMP (cm)
#   FLOW (cm)
#   28-day Compressive Strength (Mpa)

ds <- fread("./slump_test.data", header = T)
ds$No <- NULL
names(ds) <- c("cement", "slag", "fly_ash", "water", "sp", "coarse_aggr",
               "fine_aggr", "slump", "flow", "compressive_strength")

ndata <- nrow(ds)
test_size <- floor(ndata/10)

set.seed(2018)
test_idx <- sample(ndata, test_size)

ds_train = ds[-test_idx,]
ds_test = ds[test_idx,]

###########################
ds2 <- as.data.frame(ds)
# for baeirGPR
ds2$slump <- ds2$slump - mean(ds_train$slump)
mean_slump = mean(ds_train$slump)
ds2$flow <- ds2$flow - mean(ds_train$flow)
mean_flow = mean(ds_train$flow)
ds2$compressive_strength <- ds2$compressive_strength - mean(ds_train$compressive_strength)
mean_strength = mean(ds_train$compressive_strength)

ds2_train <- ds2[-test_idx,]
ds2_test <- ds2[test_idx,]

for(ii in c(1:7)) {
  ii = 1
  m1 <- mean(ds2_train[,ii])
  sd1 <- sd(ds2_train[,ii])
  ds2_train[,ii] <- (ds2_train[,ii] - m1)/sd1
  ds2_test[,ii] <- (ds2_test[,ii] - m1)/sd1
}


ds2_train_x <- model.matrix(~.-1, ds2_train[,c(1:7)])
ds2_train_y_slump <- as.matrix(ds2_train$slump)
ds2_train_y_flow <- as.matrix(ds2_train$flow)
ds2_train_y_con <- as.matrix(ds2_train$compressive_strength)


ds2_test_x <- model.matrix(~.-1, ds2_test[,c(1:7)])
ds2_test_y_slump <- as.matrix(ds2_test$slump)
ds2_test_y_flow <- as.matrix(ds2_test$flow)
ds2_test_y_con <- as.matrix(ds2_test$compressive_strength)

# t_size <- 300
# t_idx <- sample(nrow(ds_train), t_size)
tmp <- nrow(ds2_train)
ds_tune_x <- ds2_train_x[1:tmp,]
ds_tune_y <- as.matrix(ds2_train_y_slump[1:tmp,])
kern_param1 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
kern_param1 = gpr_tune(ds_tune_x, ds_tune_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50, init_betainv = kern_param1$betainv,
                       init_theta = kern_param1$thetarel)
saveRDS(kern_param1, "kern_param1.rds")
# ================================================
kern_param1 = readRDS('./kern_param1.rds')

ds2_train_yyy <- ds2_train_y_slump
# ds2_yyy <- ds2_train_y_flow
# ds2_yyy <- ds2_train_y_con
ds2_test_yyy <- ds2_test_y_slump
# ds2_test_yyy <- ds2_test_y_flow
# ds2_test_yyy <- ds2_test_y_con
#####
bsize = 50
nmodel = 300
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> gbm2_bsize_all_nmodel300.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_train_x, ds2_train_yyy, ds2_test_x, ds2_test_yyy, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE, tune_size = 20,
                                       update_kparam_tiems = update_k,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
cat(" baeirGPR rmse =", tail(gbm_model1$test_rmse))

#####
bsize = 50
nmodel = 300
update_k = 20
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 10 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> gbm3_bsizeall_nmodel300.txt &'))
t2=system.time(gbm_model2 <- gbm_train(ds2_train_x, ds2_train_yyy, ds2_test_x, ds2_test_yyy, pred_method = "gbm_sr",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = FALSE,
                                       update_kparam_tiems = update_k,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
cat(" baeirGPR rmse =", tail(gbm_model2$test_rmse))


#####
bsize = 500
nmodel = 500
update_k = 50
lr = 0.01
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 60 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> sr_bsizeall_nmodel300.txt &'))
t3=system.time(gbm_model3 <- gbm_train(ds2_train_x, ds2_train_yyy, ds2_test_x, ds2_test_y, pred_method = "gbm_sr",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))



# try ols
lm.fit <- lm(slump~.,data = ds2_train[,c(1:7,8)])
pred_ols <- predict(lm.fit, newdata=ds2_test[,c(1:7,8)])
rmse_ols <- rmse(pred_ols, ds2_test_y_con, "ols")

# try xgboost
n_feature = ncol(ds2_train_x)
tmp_xgb <- xgboost(data = data.matrix(ds2_train_x),
                   label = ds2_train_yyy,
                   eta = 0.009,
                   max_depth = 4,
                   nround = 100,
                   subsample = 0.6,
                   nthread = 4,
                   colsample_bytree = 0.9,
                   seed = 1
)
pred_xgb <- predict(tmp_xgb, ds2_test_x)
rmse_xgb <- rmse(pred_xgb, ds2_test_yyy, "xgboost")

########### other models ###########
# full gpr
gpr_model1 <- traintraintrain(ds2_train_x, ds2_train_y_slump, pred_method = "cg_direct_lm",
                              kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                              kbetainv = kern_param1$betainv, ncpu = -1, srsize = NULL,
                              clus_size = NULL)
gpr_pred1 <- gpr_fit(ds_test_x, ds_train_x, gpr_model1)
rmse_fullGPR <- rmse(gpr_pred1, ds_test_y, "full gpr")

# try kNN
pred_knn <- knn.reg(ds2_train_x, ds2_test_x, ds2_train_yyy, k = 8)$pred
rmse_knn <- rmse(pred_knn, ds2_test_yyy, "knn")

# try LASSO
mdl_lasso = cv.glmnet(ds2_train_x, ds2_train_yyy, family = "gaussian", alpha = 1, lambda = 0.5)
pred_lasso = predict(mdl_lasso, newx = ds2_test_x)
rmse_lasso = rmse(pred_lasso, ds2_test_yyy, "LASSO")

# try RIDGE
mdl_ridge = cv.glmnet(ds2_train_x, ds2_train_yyy, family = "gaussian", alpha = 0, lambda = 0.5)
pred_ridge = predict(mdl_lasso, newx = ds2_test_x)
rmse_ridge = rmse(pred_lasso, ds2_test_yyy, "RIDGE")

# try random forest
mdl_rf = ranger(slump ~ ., data = ds2_train[,c(1:7,8)], num.trees = 100, mtry = 3, write.forest = T)
pred_rf = predict(mdl_rf, ds2_test_x)
rmse(pred_rf$predictions, ds2_test_yyy, "random forest")





# ################# make file for vw ##################
ds_train_vw <- ds[-test_idx,]
ds_test_vw <- ds[test_idx]

parse_line <- function(i_row, y) {
  f_list <- paste(paste(names(i_row), i_row, sep  = ":"))
  target_value <- paste(i_row[y], "|", collapse = " ")
  feature_value <- paste(f_list[1:7], collapse = " ")
  temp <- paste(target_value, feature_value, sep = " ", collapse = "")
  return (temp)
}
# tmp = head(ds)
# tmp_test = tail(ds)

tmp = ds_train_vw
tmp_test = ds_test_vw
# for output = slump
output_form1 = apply(tmp, 1, function(x) parse_line(x, 8))
output_form_test1 = apply(tmp_test, 1, function(x) parse_line(x, 8))
# for output = flow
output_form2 = apply(tmp, 1, function(x) parse_line(x, 9))
output_form_test2 = apply(tmp_test, 1, function(x) parse_line(x, 9))
# for output = compressive_strength
output_form3 = apply(tmp, 1, function(x) parse_line(x, 10))
output_form_test3 = apply(tmp_test, 1, function(x) parse_line(x, 10))

# ##vw_train
fileConn<-file("vw_data_slump.train")
writeLines(output_form1, fileConn)
close(fileConn)
fileConn<-file("vw_data_flow.train")
writeLines(output_form2, fileConn)
close(fileConn)
fileConn<-file("vw_data_con.train")
writeLines(output_form3, fileConn)
close(fileConn)
# ##vw_test
fileConn<-file("vw_data_slump.test")
writeLines(output_form_test1, fileConn)
close(fileConn)
fileConn<-file("vw_data_flow.test")
writeLines(output_form_test2, fileConn)
close(fileConn)
fileConn<-file("vw_data_con.test")
writeLines(output_form_test3, fileConn)
close(fileConn)
# ## vw_test2
fileConn<-file("vw_data2_slump.test")
writeLines(gsub(".*\\|", "|", output_form_test1), fileConn)
close(fileConn)
fileConn<-file("vw_data2_flow.test")
writeLines(gsub(".*\\|", "|", output_form_test2), fileConn)
close(fileConn)
fileConn<-file("vw_data2_con.test")
writeLines(gsub(".*\\|", "|", output_form_test3), fileConn)
close(fileConn)

