library(data.table)

library(laGP)
#####
rmse <- function(y_hat, y, method = "") {
  out <- sqrt(mean((y - y_hat)^2))
  cat(method,"rmse = ", out, "\n")
  return(out)
}
#####
setwd('~/Desktop/gpr_testing/msd')
# step1. load dataset
ds1 <- fread('./YearPredictionMSD.txt', header = FALSE)
# load("./kern_gpr_param.rdata")
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
# ssize <- 8000
ssize <- 0
# > nrow(ds1)
# [1] 515345
# ds2 <- ds1[1:463715, ]
if(ssize > 0) {
  ds2_train <- ds2_train[1:ssize,]
}

# step2.2 matrix formulation
ds2_testmx <- model.matrix(yy~.-1, ds2_test)
ds2_test_y <- as.matrix(ds2_test$yy)
ds2_trainmx <- model.matrix(yy~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$yy)


################ mspe #######################
# t0 = Sys.time()
# p.mspe = laGP(ds2_testmx, 900, 1000, ds2_trainmx, ds2_train_y, method="mspe")
# t1 = Sys.time()
# print(t1-t0)
# rmse(p.mspe$mean, ds_test_y, "p.mspe")

# ########## alc ###################
t0 = Sys.time()
p.alc = laGP(ds2_testmx, 6, 10, ds2_trainmx, ds2_train_y, method="alc")
t1 = Sys.time()
print(t1-t0)
rmse(p.alc$mean, ds2_test_y, "alc")





