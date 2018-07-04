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
# ds <- fread("~/Desktop/gpr_testing/electricity/LD2011_2014.txt")
#
# ds = as.data.frame(ds)
# str2float = function(line) {
#   line = gsub(",", ".", line)
#   return(line)
# }


# tmp = as.data.frame(t(apply(ds, 1, str2float)))
# tmp = as.data.frame(tmp)
# setnames(tmp, c('time', paste('x', c(1:370), sep='')))

tmp = readRDS('./data.rds')


max_list = c(0)
for (i in c(2:371)) {
  max_list = c(max_list, max(as.numeric(as.character((tmp[,i])))))
}
idx_list = c(1:19)*18
# plot15min(tmp,idx_list,120000,120900)
# plot15min(tmp,idx_list,120000,120300)
plot15min<-function(data, selectColIdx, startRowIdx, endRowIdx){
  selectData = data[,selectColIdx]
  timeCol = data[,1]
  rowIdxRange = c(startRowIdx:endRowIdx)
  lebeledRowIdxs = Filter(function (x) x%%100==0, rowIdxRange)
  xLebels = timeCol[lebeledRowIdxs];
  firstSelectedCol = selectData[rowIdxRange, 1];
  plot(rowIdxRange, firstSelectedCol, type = "l", ylim = c(0, 1000), xaxt = "n")
  axis(1, at=lebeledRowIdxs, labels=xLebels)

  color = rainbow(length(selectColIdx)-1)
  for(i in c(2:length(selectColIdx))){
    otherSelectedCol = selectData[rowIdxRange, i];
    lines(rowIdxRange, otherSelectedCol, col = color[i])
  }
}


countHourTotal <- function(col){
  col = as.numeric(as.character(col))
  count = floor(length(col)/4)
  result = vector('double')
  for(i in c(0:(count-1))){
    j = i*4
    tmp = col[j+1]+col[j+2]+col[j+3]+col[j+4]
    result = c(result, tmp)
  }
  return(result)
}
tmp20 = tmp
# plot15min(tmp,idx_list,120000,120600)
# plot1hour(tmp, idx_list, 120000,120600)
plot1hour<-function(data, selectColIdx, startRowIdx, endRowIdx){
  selectData = data[,selectColIdx]
  timeCol = data[,1]
  hourCount = floor((endRowIdx-startRowIdx+1)/4)
  newEndRowIdx = startRowIdx + hourCount*4 - 1
  rowIdxRange = c(startRowIdx:newEndRowIdx)
  newRowIdxRange = c(1:hourCount)

  firstSelectedCol = countHourTotal(selectData[rowIdxRange, 1]);

  lebeledRowIdxs = (Filter(function (x) x%%(20)==0, newRowIdxRange))
  mapLebeledRowIdxs = lebeledRowIdxs * 4 + startRowIdx -1
  xLebel = timeCol[mapLebeledRowIdxs];

  plot(newRowIdxRange, firstSelectedCol, type = "l", ylim = c(0, 10000), xaxt = "n")
  axis(1, at=lebeledRowIdxs, labels=xLebel)

  color = rainbow(length(tmp20)-1)
  for(i in c(2:length(tmp20))){
    otherSelectedCol = countHourTotal(selectData[rowIdxRange, i]);
    lines(newRowIdxRange, otherSelectedCol, col = color[i])
  }
}




