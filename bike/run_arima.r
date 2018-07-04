library('ggplot2')
library('forecast')
library('tseries')

daily_data = read.csv('day.csv', header=TRUE, stringsAsFactors=FALSE)
daily_data$Date = as.Date(daily_data$dteday)
ggplot(daily_data, aes(Date, cnt)) + geom_line() + scale_x_date('month')　+
  ylab("Daily Bike Checkouts") +　xlab("")

count_ts = ts(daily_data[, c('cnt')])

# tmp1 = daily_data$cnt
# tmp2 = daily_data$clean_cnt
# ttt = vector("numeric")
# for (i in c(1:nrow(daily_data))) {
#   if (tmp1[i] != tmp2[i]) {
#     ttt = c(ttt, i)
#     print(daily_data[i,])
#   }
# }

ggplot() +
  geom_line(data = daily_data, aes(x = Date, y = clean_cnt)) + ylab('Cleaned Bicycle Count')
