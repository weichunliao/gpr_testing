library(data.table)

#load the dataset
setwd("~/Desktop/gpr_testing/msd")
ds1 <- fread('./YearPredictionMSD.txt', header = FALSE)

setnames(ds1, c('y', paste('x', 1:90, sep = '')))
ds1 <- as.data.frame(ds1)
ds1_train_vw <- ds1[1:463715, ]
ds1_test_vw <- ds1[463716:nrow(ds1),]

ds2 <- as.data.frame(ds1)
ds2$y <- (ds2$y-1922)/(2011-1922)
ds2_train_vw <- ds2[1:463715, ]
ds2_test_vw <- ds2[463716:nrow(ds1),]
###############
parse_line <- function(i_row) {
  f_list <- paste(paste(names(i_row), i_row, sep  = ":"))
  target_value <- paste(i_row[1], "|", collapse = " ")
  feature_value <- paste(f_list[2:91], collapse = " ")
  temp <- paste(target_value, feature_value, sep = " ", collapse = "")
  return (temp)
}

# tmp = ds1_train_vw
# tmp_test = ds1_test_vw
tmp = ds2_train_vw
tmp_test = ds2_test_vw
output_form = apply(tmp, 1, function(x) parse_line(x))
output_form_test = apply(tmp_test, 1, function(x) parse_line(x))

fileConn<-file("vw1_data.train")
writeLines(output_form, fileConn)
close(fileConn)
fileConn<-file("vw1_data.test")
writeLines(output_form_test, fileConn)
close(fileConn)
fileConn<-file("vw1_data_no_y.test")
writeLines(gsub(".*\\|", "|", output_form_test), fileConn)
close(fileConn)

fileConn<-file("vw2_data.train")
writeLines(output_form, fileConn)
close(fileConn)
fileConn<-file("vw2_data.test")
writeLines(output_form_test, fileConn)
close(fileConn)
fileConn<-file("vw2_data_no_y.test")
writeLines(gsub(".*\\|", "|", output_form_test), fileConn)
close(fileConn)

##################################
vw_pred <- fread("./test1.pred", header = F)
vw_rmse <- sqrt(mean((ds1_test_vw$y- vw_pred)^2))
cat(" vw rmse(w. online learning) = ", vw_rmse, "\n")

vw_pred2 <- fread("./test1_no_y.pred", header = F)
vw_rmse2 <- sqrt(mean((ds1_test_vw$y- vw_pred2)^2))
cat(" vw rmse(without online learning) = ", vw_rmse2, "\n")

vw_pred3 <- fread("./test2.pred", header = F)
vw_pred3 <- 1922+vw_pred3*(2011-1922)
vw_rmse3 <- sqrt(mean((ds1_test_vw$y- vw_pred3)^2))
cat(" vw rmse(without online learning) = ", vw_rmse3, "\n")

vw_pred4 <- fread("./test2_no_y.pred", header = F)
vw_pred4 <- 1922+vw_pred4*(2011-1922)
vw_rmse4 <- sqrt(mean((ds1_test_vw$y- vw_pred4)^2))
cat(" vw rmse(without online learning) = ", vw_rmse4, "\n")

