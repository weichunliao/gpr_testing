library(data.table)

library(xgboost)
library(FNN)
library(glmnet)
library(ranger)
#####
rmse <- function(y_hat, y, method = "") {
  out <- sqrt(mean((y - y_hat)^2))
  cat(method,"rmse = ", out, "\n")
  return(out)
}
#####
setwd("~/Desktop/gpr_testing/electricity")

# tmp = as.data.frame(t(apply(ds, 1, str2float)))
# tmp = as.data.frame(tmp)
# setnames(tmp, c('time', paste('x', c(1:370), sep='')))

tmp = readRDS('./data.rds')

#col
#1 1-140256 row
#2 year
#3 month
#4 day
#5 weekday
#6 hour
#7 quarter
#8 rowSum

weichunnnnslave<-function(data){
  resultDF = data.frame()
  j=0
  for(i in c(1:nrow(data))){
    test=as.POSIXct(data[i,1], tz = "", "%Y-%m-%d %H:%M:%S")
    year=format(test,"%Y")
    month=format(test,"%m")
    day=format(test,"%d")
    weekday= weekdays(test)
    hour=format(test,"%H")
    quarter=j%%4+1
    j=j+1
    #newRow=c(as.character(data[i,1]),year,month,day,weekday,hour,quarter)
    newRow <- data.frame(
      datetime=data[i,1],
      year=year,
      month=month,
      day=day,
      weekday=weekday,
      hour=hour,
      quarter=quarter,
      rowSum=rowSums(t(apply((data[i,2:371]), 1, as.numeric)))
    )

    resultDF = rbind(resultDF, newRow)
  }
  return(resultDF)
}

new_tmp=weichunnnnslave(tmp)
