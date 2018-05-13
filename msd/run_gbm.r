#! /usr/bin/env Rscript
# initial.options <- commandArgs(trailingOnly = FALSE)
# file.arg.name <- "--file="
# script.name <- sub(file.arg.name, "", initial.options[grep(file.arg.name, initial.options)])
# script.basename <- dirname(script.name)
# other.name <- paste(sep="/", script.basename, "other.R")
# print(script.basename)
# print(script.name)
# print(getwd())
# ==============
# ========== command line arg parsing ===========
library("optparse")

option_list = list(
  make_option(c("-m", "--method"), action="store", default=NULL, type='character',
              help="predict method"),
  make_option(c("-n", "--nmodel"), action="store", default=500, type='integer',
              help="value for n_model"),
  make_option(c("-l", "--lr"), action="store", default=0.1, type='double',
              help="set learning rate"),
  make_option(c("-b", "--batch_size"), action="store", default=3000, type = 'integer',
              help="set batch size"),
  make_option(c("-a", "--ard"), action="store_true", default=FALSE, type = 'logical',
              help="set ard to true."),
  make_option(c("-u", "--update_k"), action="store", default=50, type = 'integer',
              help="set iterations to update kernel param")
)
opt = parse_args(OptionParser(option_list=option_list))
# print(opt)

# =========== load dataset and run gbm ===========
library(data.table)

#load the dataset
# setwd('~/gpr_testing/')
ds1 <- fread('./YearPredictionMSD.txt', header = FALSE)
load("./kern_gpr_param.rdata")
# ds1[1:5,1:5]

setnames(ds1, c('yy', paste('x', 1:90, sep = '')))
ds1 <- as.data.frame(ds1)

#normalize to mean 0 and unit variance.
ds1$yy <- ds1$yy - mean(ds1$yy)

#random permutation
#There are 463715 training records
# ds2_train = ds1[1:463715, ]
# ds2_test = ds1[463716:nrow(ds1),]
# ds2_train <- ds2[sample(nrow(ds2_train)),]
#
for(ii in 2:ncol(ds1)) {
  m1 <- mean(ds1[,ii])
  sd1 <- sd(ds1[,ii])
  ds1[,ii] <- (ds1[,ii] - m1)/sd1
}

#random permutation
#There are 463715 training records
ds2 <- ds1[1:463715, ]
ds2_test <- ds1[463716:nrow(ds1),]
ds2 <- ds2[sample(nrow(ds2)),]

#the size of training data
# ssize <- 460000
# ssize <- 200
ssize <- 0
# > nrow(ds1)
# [1] 515345
# ds2 <- ds1[1:463715, ]

if(ssize == 0) {
  ds2_train <- ds2
} else {
  ds2_train <- ds2[1:ssize,]
}

ds2_testmx <- model.matrix(yy~.-1, ds2_test)
ds2_test_y <- as.matrix(ds2_test$yy)
ds2_trainmx <- model.matrix(yy~.-1, ds2_train)
ds2_train_y <- as.matrix(ds2_train$yy)

library("baeirGPR", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.4")

user_pred_method = opt$method
user_n_model = opt$nmodel
user_lr = opt$lr
user_batch_size = opt$batch_size
user_tune_param = opt$ard
user_update_kparam = opt$update_k



t1=system.time(gbm_model1 <- gbm_train(ds2_trainmx, ds2_train_y, ds2_testmx, ds2_test_y, pred_method = user_pred_method,
                                       n_model = user_n_model, batch_size = user_batch_size, lr = user_lr,
                                       tune_param = user_tune_param, update_kparam_tiems = user_update_kparam,
                                       update_col_sample = user_update_kparam,
                                       kname = "gaussiandotrel", ktheta = kern_param1$thetarel,
                                       kbetainv = kern_param1$betainv))

arg_list = paste(unlist(opt[1:6]), collapse = '_')

out_fname = paste("gbm_", arg_list, ".rdata", sep="")

# saveRDS(object = gbm_model1, file = out_fname)
save(t1, gbm_model1, opt, file=out_fname)
