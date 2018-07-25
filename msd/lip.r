library(data.table)
library(baeirGPR)
###########
rmse <- function(y_hat, y, method = "") {
  out <- sqrt(mean((y - y_hat)^2))
  cat(method,"rmse = ", out, "\n")
  return(out)
}
#####

setwd("~/gpr_testing/msd")
#the size of training data
ssize <- 500

# step1. load dataset
ds1 <- fread('./YearPredictionMSD.txt', header = FALSE)
# load("./kern_gpr_param.rdata")
kern_param1 = readRDS('./kern_param2.rds')
# ds1[1:5,1:5]

setnames(ds1, c('yy', paste('x', 1:90, sep = '')))
ds1 <- as.data.frame(ds1)

# normalize to mean 0 and unit variance.
mean_shift = mean(ds1$yy)
ds1$yy <- ds1$yy - mean(ds1$yy)
ds1_train = ds1[1:463715, ]
# ds1_test = ds1[463716:nrow(ds1),]

ds2 = ds1
for(ii in 2:ncol(ds1)) {
  m1 <- mean(ds1_train[,ii])
  sd1 <- sd(ds1_train[,ii])
  ds2[,ii] <- (ds2[,ii] - m1)/sd1
}

# step2. train-test split
# random permutation
# There are 463715 training records
ds2_train = ds2[1:463715, ]
ds2_test = ds2[463716:nrow(ds1),]
ds2_train <- ds2_train[sample(nrow(ds2_train)),]

# step2.1 set the size of training data
# ssize <- 460000
ssize <- 50000
# ssize <- 0
# > nrow(ds1)
# [1] 515345
# ds2 <- ds1[1:463715, ]
if(ssize > 0) {
  ds2_train <- ds2_train[1:ssize,]
}

ds2_testmx <- model.matrix(yy~.-1, ds2_test)
ds2_test_y <- as.matrix(ds2_test$yy)
ds2_trainmx <- model.matrix(yy~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$yy)

####################### run models ################################
session_pid = Sys.getpid()
cmd_arg = paste('pidstat \\-r \\-t 60 \\-p', session_pid, sep = ' ')
system(paste(cmd_arg, '> cg_direct_lm_50000.txt &'))
t0 = Sys.time()
gpr_model3 <- traintraintrain(ds2_trainmx, ds2_train_y, pred_method = "cg_direct_lm",
                              kname = kern_param1$kernelname, ktheta = kern_param1$thetarel, kbetainv = kern_param1$betainv,
                              ncpu = -1)
# gpr_model4 <- traintraintrain(ds2_trainmx, ds2_train_y, pred_method = "usebigK",
#                               kname = kern_param1$kernelname, ktheta = kern_param1$thetarel, kbetainv = kern_param1$betainv,
#                               ncpu = -1, tsize = 100)
# gpr_model5 <- traintraintrain(ds2_trainmx, ds2_train_y, pred_method = "local_gpr",
#                               kname = kern_param1$kernelname, ktheta = kern_param1$thetarel, kbetainv = kern_param1$betainv,
#                               ncpu = -1, tsize = 100, clus_size = 50)
# gpr_model6 <- traintraintrain(ds2_trainmx, ds2_train_y, pred_method = "sr",
#                               kname = kern_param1$kernelname, ktheta = kern_param1$thetarel, kbetainv = kern_param1$betainv,
#                               ncpu = -1, srsize = 5000)

predict3 <- gpr_fit(ds2_testmx, ds2_trainmx, gpr_model3)
rmse(ds2_test_y, predict3)
t1 = Sys.time()
system(paste("kill $(ps aux | grep -i '", cmd_arg ,"' | awk -F' ' '{ print $2 }')", sep=''))
print(t1-t0)
# predict4 <- gpr_fit(ds2_testmx, ds2_trainmx, gpr_model4)
# rmse(ds2_test_y, predict4)
# predict5 <- gpr_fit(ds2_testmx, ds2_trainmx, gpr_model5)
# rmse(ds2_test_y, predict5)
# predict6 <- gpr_fit(ds2_testmx, ds2_trainmx, gpr_model6)
# rmse(ds2_test_y, predict6)

