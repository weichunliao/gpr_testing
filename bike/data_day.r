library(data.table)
library(dummies)

library(ranger)

library(baeirGPR)
######
setwd('~/Desktop/gpr_testing/bike/')
tmp = fread('./day.csv')
tmp = as.data.frame(tmp)

feature_names <- c("instant", "season", "yr", "mnth", "holiday", "weekday", "workingday",
                   "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt")
tmp2 = tmp[,feature_names]

dummy_names <- c("season", "weekday", "weathersit")
ds_dummy <- dummy.data.frame(tmp2, names = dummy_names)
dummy_cnames = names(ds_dummy)
ds_dummy <- as.data.frame(apply(ds_dummy, 2, as.numeric))
colnames(ds_dummy) <- dummy_cnames

ndata = nrow(ds_dummy)
ds = cbind(ds_dummy[2:ndata,1:16], ds_dummy[1:(ndata-1), 17:25], ds_dummy[2:ndata,26])
colnames(ds) = dummy_cnames

test_size <- floor(ndata/10)
train_idx <- ndata-test_size-1
ds_train = ds[1:train_idx,]
ds_test = ds[(train_idx+1):nrow(ds),]

ds2 = ds
mean_shift = mean(ds_train$cnt)
ds2$cnt = ds2$cnt - mean_shift

ds2_train = ds2[1:train_idx,]
ds2_test = ds2[(train_idx+1):nrow(ds),]

for (i in c(20:25)){
  m1 <- mean(ds2_train[,i])
  sd1 <- sd(ds2_train[,i])
  ds2_train[,i] <- (ds2_train[,i] - m1)/sd1
  ds2_test[,i] <- (ds2_test[,i] - m1)/sd1
}

ds2_train_x <- model.matrix(cnt~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$cnt)
ds2_test_x <- model.matrix(cnt~.-1, ds2_test)
ds2_test_y <- as.matrix(ds2_test$cnt)

#### tune kern_param
kern_param1 = gpr_tune(ds2_train_x, ds2_train_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
kern_param1 = gpr_tune(ds2_train_x, ds2_train_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50, init_betainv = kern_param1$betainv,
                       init_theta = kern_param1$thetarel)
saveRDS(kern_param1, "kern_param1.rds")
## noard
kern_param1n = gpr_tune(ds2_train_x, ds2_train_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50)
kern_param1n = gpr_tune(ds2_train_x, ds2_train_y, kernelname = "rbf",
                       optim_report = 5, optim_ard_report = 5,
                       ARD = T, optim_ard_max = 50, init_betainv = kern_param1n$betainv,
                       init_theta = kern_param1n$thetarel)
saveRDS(kern_param1n, "kern_param1n.rds")


###
kern_param1 = readRDS('./kern_param1.rds')
bsize = 1000
nmodel = 10
update_k = 50
lr = 0.1
# session_pid = Sys.getpid()
# cmd_arg = paste('pidstat \\-r \\-t 60 \\-p', session_pid, sep = ' ')
# system(paste(cmd_arg, '> gbm2_bsize4000_nmodel500.txt &'))
t1=system.time(gbm_model1 <- gbm_train(ds2_train_x, ds2_train_y, ds2_test_x, ds2_test_y, pred_method = "2",
                                       n_model = nmodel, batch_size = bsize, lr = lr, tune_param = TRUE,
                                       update_kparam_tiems = update_k, decay_lr = 0.9,
                                       kname = kern_param1$kernelname, ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))
# system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))




