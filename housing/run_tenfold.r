library(data.table)
require(caret)

library(xgboost)
library(FNN)
library(glmnet)
library(ranger)
library(e1071)

library(baeirGPR)
#########
rmse <- function(y_hat, y, method = "") {
  out <- sqrt(mean((y - y_hat)^2))
  cat(method,"rmse = ", out, "\n")
  return(out)
}
########

setwd("~/Desktop/gpr_testing/housing")
ds <- fread("./housing.data")

feature_names <- c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
                   "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV")
ds <- setNames(ds, feature_names)
ds <- as.data.frame(apply(ds, 2, as.numeric))

ndata <- nrow(ds)

set.seed(2018)
# k-folds
k = 10
flds <- createFolds(c(1:ndata), k = k, list = TRUE, returnTrain = FALSE)

results = vector("list", k)
rmse_list = vector('numeric')

# method = "gbm2"
# method = "gbm3"
method = "gbm_sr"
# method = "ols"
# method = "losso"
# method = "ridge"
# method = "knn"
# method = "xgboost"
# method = "rf"
# method = "svr_linear"
# method = "svr_poly"
# method = "svr_rbf"


for (kfld in c(1:k)) {
  # kfld = 1#####################
  k_test_idx = flds[[kfld]]
  k_ds_train = ds[-k_test_idx,]
  k_ds_test = ds[k_test_idx,]

  k_ds2 = ds
  # step1. mean shift
  mean_shift = mean(k_ds_train$MEDV)
  k_ds2$MEDV = k_ds2$MEDV - mean_shift

  # step2. normalize non-dummy variable
  for (i in c(1:3, 5:13)){
    m_i = mean(k_ds_train[, i])
    std_i = sd(k_ds_train[, i])

    k_ds2[,i] = (k_ds2[,i] - m_i)/std_i
  }
  # step3. train-test split
  k_ds2_train = k_ds2[-k_test_idx,]
  k_ds2_train_x = model.matrix(MEDV~.-1, k_ds2_train)
  k_ds2_train_y <- as.matrix(k_ds2_train$MEDV)

  k_ds2_test <- k_ds2[k_test_idx,]
  k_ds2_test_x <- model.matrix(MEDV~.-1, k_ds2_test)
  k_ds2_test_y <- as.matrix(k_ds2_test$MEDV)

  if (method == "gbm2") {
    kern_param1 = readRDS('./kern_param1.rds')
    bsize = 100
    nmodel = 300
    update_k = 20
    t1=system.time(gbm_model1 <- gbm_train(k_ds2_train_x, k_ds2_train_y, k_ds2_test_x, k_ds2_test_y,
                                           pred_method = "2",
                                           n_model = nmodel, batch_size = bsize, lr = 0.05, tune_param = TRUE,
                                           update_kparam_tiems = update_k, decay_lr = 0.9, update_lr = 40,
                                           kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                           kbetainv = kern_param1$betainv))
    results[[kfld]] = gbm_model1$test_rmse
  } else if (method == 'gbm3') {
    kern_param1 = readRDS('./kern_param1.rds')
    bsize = 100
    nmodel = 300
    update_k = 20
    t2=system.time(gbm_model2 <- gbm_train(k_ds2_train_x, k_ds2_train_y, k_ds2_test_x, k_ds2_test_y,
                                           pred_method = "3",
                                           n_model = nmodel, batch_size = bsize, lr = 0.05, tune_param = FALSE,
                                           update_kparam_tiems = update_k,
                                           kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                           kbetainv = kern_param1$betainv))
    results[[kfld]] = gbm_model2$test_rmse
  } else if (method == 'gbm_sr') {
    kern_param1 = readRDS('./kern_param1.rds')
    bsize = 225
    nmodel = 300
    update_k = 20
    lr = 0.1
    t3=system.time(gbm_model3 <- gbm_train(k_ds2_train_x, k_ds2_train_y, k_ds2_test_x, k_ds2_test_y,
                                           pred_method = "gbm_sr",
                                           n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                           update_kparam_tiems = update_k,
                                           kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                           kbetainv = kern_param1$betainv))
    results[[kfld]] = gbm_model3$test_rmse
  } else if (method == 'ols') {
    lm.fit <- lm(MEDV~.,data = k_ds2_train)
    pred_ols <- predict(lm.fit, newdata = k_ds2_test)
    rmse_ols <- rmse(pred_ols, k_ds2_test_y, "ols")

    results[[kfld]] = rmse_ols
  } else if (method == "losso") {
    mdl_lasso = cv.glmnet(k_ds2_train_x, k_ds2_train_y, family = "gaussian", alpha = 1)
    pred_lasso = predict(mdl_lasso, newx = k_ds2_test_x)
    rmse_losso = rmse(pred_lasso, k_ds2_test_y, "LASSO")

    results[[kfld]] = rmse_losso
  } else if (method == "ridge") {
    lambdas <- 10^seq(50, -50, by = -.01)
    mdl_ridge = cv.glmnet(k_ds2_train_x, k_ds2_train_y, family = "gaussian", alpha = 0, lambda = lambdas)
    pred_ridge = predict(mdl_ridge, newx = k_ds2_test_x)
    rmse_ridge = rmse(pred_ridge, k_ds2_test_y, "RIDGE")

    results[[kfld]] = rmse_ridge
  } else if (method == "knn") {
    pred_knn1 <- knn.reg(k_ds2_train_x, k_ds2_test_x, k_ds2_train_y, k = 3)$pred
    rmse_knn <- rmse(pred_knn1, k_ds2_test_y, "knn")

    results[[kfld]] = rmse_knn
  } else if (method == "xgboost") {
    n_feature = ncol(k_ds_train)
    tmp_xgb <- xgboost(data = data.matrix(k_ds2_train_x),
                       label = k_ds2_train_y,
                       eta = 0.05,
                       max_depth = 6,
                       nround = 200,
                       subsample = 0.7,
                       nthread = 4,
                       colsample_bytree = 1,
                       seed = 1
    )
    pred_xgb <- predict(tmp_xgb, k_ds2_test_x)
    rmse_xgb <- rmse(pred_xgb, k_ds2_test_y, "xgboost")

    results[[kfld]] = rmse_xgb
  } else if (method == "rf") {
    mdl_svr = svm(MEDV~., k_ds2_train, kernel = "linear")
    # mdl_rf = ranger(MEDV ~ ., data = k_ds2_train, num.trees = 100, mtry = 5, write.forest = T)
    pred_rf = predict(mdl_rf, k_ds2_test_x)
    rmse_rf = rmse(pred_rf$predictions, k_ds2_test_y, "random forest")

    results[[kfld]] = rmse_rf
  } else if (method == "svr_linear") {
    mdl_svr = svm(MEDV~., k_ds2_train, kernel = "linear")
    svr.pred = predict(mdl_svr, k_ds2_test_x)
    rmse_svr = rmse(svr.pred, k_ds2_test_y, "svr")

    results[[kfld]] = rmse_svr
  } else if (method == "svr_poly") {
    mdl_svr = svm(MEDV~., k_ds2_train, kernel = "polynomial", degree = 3)
    svr.pred = predict(mdl_svr, k_ds2_test_x)
    rmse_svr = rmse(svr.pred, k_ds2_test_y, "svr")


    results[[kfld]] = rmse_svr
  } else if (method == "svr_rbf") {
    mdl_svr = tune.svm(MEDV~., data = k_ds2_train, kernel = "radial", gamma = 2^c(-8:0), cost = 2^c(-2:6))
    svr.pred = predict(mdl_svr$best.model, k_ds2_test_x)
    rmse_svr = rmse(svr.pred, k_ds2_test_y, "svr")

    results[[kfld]] = rmse_svr
  }

} # end for

n_round = 1
if (method %in% c("ols", "losso", "ridge", "knn", "xgboost", "rf", "svr_linear", "svr_poly", "svr_rbf")) {
  all_rmse = unlist(results)
} else if (method %in% c("gbm2", "gbm3", "gbm_sr")) {
  all_rmse = numeric()
  for (kfld in c(1:k)) {
    all_rmse = c(all_rmse, results[[kfld]][n_round])
  }
}
cat(mean(all_rmse), "\n")

