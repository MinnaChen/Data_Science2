library(ggpubr)
library(ggplot2)
library(e1071)
library(dplyr) #add rows
library(lmridge) #for ridge regression
library(lars) # lasso regression
library(caret) # for cross validation
library(gdata) #to read xls file format
library(rsm) # for response surface


df <- read.csv('/home/mr_malviya/Desktop/Data_Science_2/ls')
conc_df <- read.xls('/home/mr_malviya/Desktop/College/Data_Science_2/Datasets/Concrete/Concrete_Data.xls')


#checking the structure of the df
str(df)

#making a copy of the original dataframe
auto_df <- df

#replacing the missing values by mean

#auto mpg dataset
auto_df$cylinders[is.na(auto_df$cylinders)] <- mean(auto_df$cylinders, na.rm = TRUE)
auto_df$displacement[is.na(auto_df$displacement)] <- mean(auto_df$displacement, na.rm = TRUE)
auto_df$weight[is.na(auto_df$weight)] <- mean(auto_df$weight, na.rm = TRUE)
auto_df$acceleration[is.na(auto_df$acceleration)] <- mean(auto_df$acceleration, na.rm = TRUE)
auto_df$model.year[is.na(auto_df$model.year)] <- mean(auto_df$model.year, na.rm = TRUE)
auto_df$origin[is.na(auto_df$origin)] <- mean(auto_df$origin, na.rm = TRUE)

#concrete dataset

#modifying the column names
names(conc_df) <- c("Cement", "Slag", "Flyash", "Water", "Plasticizer", "CAggregate", "FAggregate", "Age", "CCStrength")
conc_df$Cement[is.na(conc_df$Cement)] <- mean(conc_df$Cement, na.rm = TRUE)
conc_df$Slag[is.na(conc_df$Slag)] <- mean(conc_df$Slag, na.rm = TRUE)
conc_df$Flyash[is.na(conc_df$Flyash)] <- mean(conc_df$Flyash, na.rm = TRUE)
conc_df$Water[is.na(conc_df$Water)] <- mean(conc_df$Water, na.rm = TRUE)
conc_df$Plasticizer[is.na(conc_df$Plasticizer)] <- mean(conc_df$Plasticizer, na.rm = TRUE)
conc_df$CAggregate[is.na(conc_df$CAggregate)] <- mean(conc_df$CAggregate, na.rm = TRUE)
conc_df$FAggregate[is.na(conc_df$FAggregate)] <- mean(conc_df$FAggregate, na.rm = TRUE)
conc_df$Age[is.na(conc_df$Age)] <- mean(conc_df$Age, na.rm = TRUE)
conc_df$CCStrength[is.na(conc_df$CCStrength)] <- mean(conc_df$CCStrength, na.rm = TRUE)




#converting the horsepower into numeric
auto_df$horsepower <- as.numeric(as.character(df$horsepower))
auto_df$horsepower[is.na(auto_df$horsepower)] <- mean(auto_df$horsepower, na.rm = TRUE)

#removing car names from the data
auto_df <- auto_df[c(-9)]


### Getting scatter plots of each of the variables 
scatter.smooth(x = df$displacement, y = df$mpg, main = 'mpg vs displacement')
scatter.smooth(x = df$horsepower, y = df$mpg, main = 'mpg vs horsepower')
scatter.smooth(x = df$cylinders, y = df$mpg, main = 'mpg vs cylinders')
scatter.smooth(x = df$weight, y = df$mpg, main = 'mpg vs weight')
scatter.smooth(x = df$accerleration, y = df$mpg, main = 'mpg vs acceleration')
scatter.smooth(x = df$model.year, y = df$mpg, main = 'mpg vs model.year')


##forward selection 

#auto mpg dataset
fwd_model_auto <- lm(mpg ~ 1,  data = auto_df) 
step(fwd_model_auto, direction = "forward", scope = formula(mpg ~ cylinders + displacement + horsepower + weight + acceleration + model.year + origin))
summary(fwd_model_auto)
## features obtained from forward selection mpg ~ weight + model.year + horsepower + origin + acceleration

#concrete dataset
fwd_model_conc <- lm(CCStrength ~ 1,  data = auto_df) 
step(fwd_model_conc, direction = "forward", scope = formula(CCStrength ~ Cement + Slag + Flyash + Water + Plasticizer + CAggregate + FAggregate + Age))
summary(fwd_model_conc)
## features obtained from forward selection  CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash



#specifying method for crossvalidation 
train_control <- trainControl(method = "cv", number = 10)

####LINEAR REGRESSION

#auto mpg
lin_mod1 <- lm(mpg ~ weight, data = auto_df)
lin_mod2 <- lm(mpg ~ weight + model.year, data = auto_df)
lin_mod3 <- lm(mpg ~ weight + model.year + horsepower, data = auto_df)
lin_mod4 <- lm(mpg ~ weight + model.year + horsepower + origin, data = auto_df)
lin_mod5 <- lm(mpg ~ weight + model.year + horsepower + origin + acceleration, data = auto_df)
lin_mod6 <- lm(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders, data = auto_df)
lin_mod7 <- lm(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders + displacement, data = auto_df)

#conc data
lin_conc_mod1 <- lm(CCStrength ~ Cement, data = conc_df)
lin_conc_mod2 <- lm(CCStrength ~ Cement + Plasticizer , data = conc_df)
lin_conc_mod3 <- lm(CCStrength ~ Cement + Plasticizer + Age, data = conc_df)
lin_conc_mod4 <- lm(CCStrength ~ Cement + Plasticizer + Age + Slag, data = conc_df)
lin_conc_mod5 <- lm(CCStrength ~ Cement + Plasticizer + Age + Slag + Water, data = conc_df)
lin_conc_mod6 <- lm(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash, data = conc_df)
lin_conc_mod7 <- lm(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash + FAggregate, data = conc_df)
lin_conc_mod8 <- lm(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash + FAggregate + CAggregate, data = conc_df)

#cross-validation from here
#auto mpg
cv_model1 <- train(mpg ~ weight, data = auto_df, trControl = train_control, method = "lm")
cv_model2 <- train(mpg ~ weight + model.year, data = auto_df, trControl = train_control, method = "lm")
cv_model3 <- train(mpg ~ weight + model.year + horsepower, data = auto_df, trControl = train_control, method = "lm")
cv_model4 <- train(mpg ~ weight + model.year + horsepower + origin, data = auto_df, trControl = train_control, method = "lm")
cv_model5 <- train(mpg ~ weight + model.year + horsepower + origin + acceleration, data = auto_df, trControl = train_control, method = "lm")
cv_model6 <- train(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders, data = auto_df, trControl = train_control, method = "lm")
cv_model7 <- train(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders + displacement, data = auto_df, trControl = train_control, method = "lm")

#conc
cv_conc_model1 <- train(CCStrength ~ Cement, data = conc_df, trControl = train_control, method = "lm")
cv_conc_model2 <- train(CCStrength ~ Cement + Plasticizer, data = conc_df, trControl = train_control, method = "lm")
cv_conc_model3 <- train(CCStrength ~ Cement + Plasticizer + Age, data = conc_df, trControl = train_control, method = "lm")
cv_conc_model4 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag, data = conc_df, trControl = train_control, method = "lm")
cv_conc_model5 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag + Water, data = conc_df, trControl = train_control, method = "lm")
cv_conc_model6 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash, data = conc_df, trControl = train_control, method = "lm")
cv_conc_model7 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash + FAggregate, data = conc_df, trControl = train_control, method = "lm")
cv_conc_model8 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash + FAggregate + CAggregate, data = conc_df, trControl = train_control, method = "lm")



#creating a new dataframe to store rsquare adjusted_rsquard and cv rsquared

#auto mpg
error_df_lm <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_df_lm <- add_row(error_df_lm, r_sq = summary(lin_mod1)$r.squared, adj_r_sq = summary(lin_mod1)$adj.r.squared, cv_r_sq = mean(cv_model1$resample$Rsquared))
error_df_lm <- add_row(error_df_lm, r_sq = summary(lin_mod2)$r.squared, adj_r_sq = summary(lin_mod2)$adj.r.squared, cv_r_sq = mean(cv_model2$resample$Rsquared))
error_df_lm <- add_row(error_df_lm, r_sq = summary(lin_mod3)$r.squared, adj_r_sq = summary(lin_mod3)$adj.r.squared, cv_r_sq = mean(cv_model3$resample$Rsquared))
error_df_lm <- add_row(error_df_lm, r_sq = summary(lin_mod4)$r.squared, adj_r_sq = summary(lin_mod4)$adj.r.squared, cv_r_sq = mean(cv_model4$resample$Rsquared))
error_df_lm <- add_row(error_df_lm, r_sq = summary(lin_mod5)$r.squared, adj_r_sq = summary(lin_mod5)$adj.r.squared, cv_r_sq = mean(cv_model5$resample$Rsquared))
error_df_lm <- add_row(error_df_lm, r_sq = summary(lin_mod6)$r.squared, adj_r_sq = summary(lin_mod6)$adj.r.squared, cv_r_sq = mean(cv_model6$resample$Rsquared))
error_df_lm <- add_row(error_df_lm, r_sq = summary(lin_mod7)$r.squared, adj_r_sq = summary(lin_mod7)$adj.r.squared, cv_r_sq = mean(cv_model7$resample$Rsquared))

#conc data
error_conc_lm <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_conc_lm <- add_row(error_conc_lm, r_sq = summary(lin_conc_mod1)$r.squared, adj_r_sq = summary(lin_conc_mod1)$adj.r.squared, cv_r_sq = mean(cv_conc_model1$resample$Rsquared))
error_conc_lm <- add_row(error_conc_lm, r_sq = summary(lin_conc_mod2)$r.squared, adj_r_sq = summary(lin_conc_mod2)$adj.r.squared, cv_r_sq = mean(cv_conc_model2$resample$Rsquared))
error_conc_lm <- add_row(error_conc_lm, r_sq = summary(lin_conc_mod3)$r.squared, adj_r_sq = summary(lin_conc_mod3)$adj.r.squared, cv_r_sq = mean(cv_conc_model3$resample$Rsquared))
error_conc_lm <- add_row(error_conc_lm, r_sq = summary(lin_conc_mod4)$r.squared, adj_r_sq = summary(lin_conc_mod4)$adj.r.squared, cv_r_sq = mean(cv_conc_model4$resample$Rsquared))
error_conc_lm <- add_row(error_conc_lm, r_sq = summary(lin_conc_mod5)$r.squared, adj_r_sq = summary(lin_conc_mod5)$adj.r.squared, cv_r_sq = mean(cv_conc_model5$resample$Rsquared))
error_conc_lm <- add_row(error_conc_lm, r_sq = summary(lin_conc_mod6)$r.squared, adj_r_sq = summary(lin_conc_mod6)$adj.r.squared, cv_r_sq = mean(cv_conc_model6$resample$Rsquared))
error_conc_lm <- add_row(error_conc_lm, r_sq = summary(lin_conc_mod7)$r.squared, adj_r_sq = summary(lin_conc_mod7)$adj.r.squared, cv_r_sq = mean(cv_conc_model7$resample$Rsquared))
error_conc_lm <- add_row(error_conc_lm, r_sq = summary(lin_conc_mod8)$r.squared, adj_r_sq = summary(lin_conc_mod8)$adj.r.squared, cv_r_sq = mean(cv_conc_model8$resample$Rsquared))

#plot of r square, adjusted r square and r square cross-validation
plot(error_df$r_sq, type = 'l', col = 'red', main = "ERROR PLOT", ylab = "Errors", ylim = c(0.7,1))
lines(error_df$adj_r_sq,  col = 'green' )
lines(error_df$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)

plot(error_conc_lm$r_sq, type = 'l', col = 'red', main = "ERROR PLOT LINEAR", ylab = "Errors", ylim = c(0.7,1))
lines(error_conc_lm$adj_r_sq,  col = 'green' )
lines(error_conc_lm$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)



#### WEIGHTED LEAST SQUARE REGRESSION

#auto mpg
wls_mod1 <- lm(mpg ~ weight, data = auto_df, weight = 1/auto_df$displacement)
wls_mod2 <- lm(mpg ~ weight + model.year, data = auto_df, weight = 1/auto_df$displacement)
wls_mod3 <- lm(mpg ~ weight + model.year + horsepower, data = auto_df, weight = 1/auto_df$displacement)
wls_mod4 <- lm(mpg ~ weight + model.year + horsepower + origin, data = auto_df, weight = 1/auto_df$displacement)
wls_mod5 <- lm(mpg ~ weight + model.year + horsepower + origin + acceleration, data = auto_df, weight = 1/auto_df$displacement)
wls_mod6 <- lm(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders, data = auto_df, weight = 1/auto_df$displacement)
wls_mod7 <- lm(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders + displacement, data = auto_df,weight = 1/auto_df$displacement)


error_df_wls <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0))
error_df_wls <- add_row(error_df_wls, r_sq = summary(wls_mod1)$r.squared, adj_r_sq = summary(wls_mod1)$adj.r.squared)
error_df_wls <- add_row(error_df_wls, r_sq = summary(wls_mod2)$r.squared, adj_r_sq = summary(wls_mod2)$adj.r.squared)
error_df_wls <- add_row(error_df_wls, r_sq = summary(wls_mod3)$r.squared, adj_r_sq = summary(wls_mod3)$adj.r.squared)
error_df_wls <- add_row(error_df_wls, r_sq = summary(wls_mod4)$r.squared, adj_r_sq = summary(wls_mod4)$adj.r.squared)
error_df_wls <- add_row(error_df_wls, r_sq = summary(wls_mod5)$r.squared, adj_r_sq = summary(wls_mod5)$adj.r.squared)
error_df_wls <- add_row(error_df_wls, r_sq = summary(wls_mod6)$r.squared, adj_r_sq = summary(wls_mod6)$adj.r.squared)
error_df_wls <- add_row(error_df_wls, r_sq = summary(wls_mod7)$r.squared, adj_r_sq = summary(wls_mod7)$adj.r.squared)


#plot of r square, adjusted r square 
plot(error_df_wls$r_sq, type = 'l', col = 'red', main = "ERROR PLOT", ylab = "Errors", ylim = c(0.3,1))
lines(error_df_wls$adj_r_sq,  col = 'green' )
legend(5,0.8, legend = c("R Squared","Adj R Squared"), 
       col = c("red","green"), lty = 1:2, cex = 0.8)


###RIDGE REGRESSION

#auto mpg
ridge_mod1 <- lmridge(mpg ~ weight + model.year, auto_df, K = c(0.1, 0.001))
ridge_mod2 <- lmridge(mpg ~ weight + model.year + horsepower, auto_df, K = c(0.1, 0.001))
ridge_mod3 <- lmridge(mpg ~ weight + model.year + horsepower + origin, auto_df, K = c(0.1, 0.001))
ridge_mod4 <- lmridge(mpg ~ weight + model.year + horsepower + origin + acceleration , auto_df, K = c(0.1, 0.001))
ridge_mod5 <- lmridge(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders, auto_df, K = c(0.1, 0.001))
ridge_mod6 <- lmridge(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders + displacement, auto_df, K = c(0.1, 0.001))

#conc data
ridge_conc_mod1 <- lmridge(CCStrength ~ Cement + Plasticizer, conc_df, K = c(0.1, 0.001))
ridge_conc_mod2 <- lmridge(CCStrength ~ Cement + Plasticizer + Age, conc_df, K = c(0.1, 0.001))
ridge_conc_mod3 <- lmridge(CCStrength ~ Cement + Plasticizer + Age + Slag, conc_df, K = c(0.1, 0.001))
ridge_conc_mod4 <- lmridge(CCStrength ~ Cement + Plasticizer + Age + Slag + Water , conc_df, K = c(0.1, 0.001))
ridge_conc_mod5 <- lmridge(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash, conc_df, K = c(0.1, 0.001))
ridge_conc_mod6 <- lmridge(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash + FAggregate, conc_df, K = c(0.1, 0.001))
ridge_conc_mod7 <- lmridge(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash + FAggregate + CAggregate, conc_df, K = c(0.1, 0.001))




#cross-validation from here

#auto mpg
train_control <- trainControl(method = "cv", number = 10)
cv_ridge_model1 <- train(mpg ~ weight + model.year, data = auto_df, trControl = train_control, method = "ridge")
cv_ridge_model2 <- train(mpg ~ weight + model.year + horsepower, data = auto_df, trControl = train_control, method = "ridge")
cv_ridge_model3 <- train(mpg ~ weight + model.year + horsepower + origin, data = auto_df, trControl = train_control, method = "ridge")
cv_ridge_model4 <- train(mpg ~ weight + model.year + horsepower + origin + acceleration, data = auto_df, trControl = train_control, method = "ridge")
cv_ridge_model5 <- train(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders, data = auto_df, trControl = train_control, method = "ridge")
cv_ridge_model6 <- train(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders + displacement, data = auto_df, trControl = train_control, method = "ridge")

#conc data
cv_ridge_conc1 <- train(CCStrength ~ Cement + Plasticizer, data = conc_df, trControl = train_control, method = "ridge")
cv_ridge_conc2 <- train(CCStrength ~ Cement + Plasticizer + Age, data = conc_df, trControl = train_control, method = "ridge")
cv_ridge_conc3 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag, data = conc_df, trControl = train_control, method = "ridge")
cv_ridge_conc4 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag + Water, data = conc_df, trControl = train_control, method = "ridge")
cv_ridge_conc5 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash, data = conc_df, trControl = train_control, method = "ridge")
cv_ridge_conc6 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash + FAggregate, data = conc_df, trControl = train_control, method = "ridge")
cv_ridge_conc7 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash + FAggregate + CAggregate, data = conc_df, trControl = train_control, method = "ridge")

#auto mpg
error_df_ridge <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_df_ridge <- add_row(error_df_ridge, r_sq = max(rstats1(ridge_mod1)$R2), adj_r_sq = max(rstats1(ridge_mod1)$adjR2), cv_r_sq = mean(cv_ridge_model1$resample$Rsquared))
error_df_ridge <- add_row(error_df_ridge, r_sq = max(rstats1(ridge_mod2)$R2), adj_r_sq = max(rstats1(ridge_mod2)$adjR2), cv_r_sq = mean(cv_ridge_model2$resample$Rsquared))
error_df_ridge <- add_row(error_df_ridge, r_sq = max(rstats1(ridge_mod3)$R2), adj_r_sq = max(rstats1(ridge_mod3)$adjR2), cv_r_sq = mean(cv_ridge_model3$resample$Rsquared))  
error_df_ridge <- add_row(error_df_ridge, r_sq = max(rstats1(ridge_mod4)$R2), adj_r_sq = max(rstats1(ridge_mod4)$adjR2), cv_r_sq = mean(cv_ridge_model4$resample$Rsquared))
error_df_ridge <- add_row(error_df_ridge, r_sq = max(rstats1(ridge_mod5)$R2), adj_r_sq = max(rstats1(ridge_mod5)$adjR2), cv_r_sq = mean(cv_ridge_model5$resample$Rsquared))
error_df_ridge <- add_row(error_df_ridge, r_sq = max(rstats1(ridge_mod6)$R2), adj_r_sq = max(rstats1(ridge_mod6)$adjR2), cv_r_sq = mean(cv_ridge_model6$resample$Rsquared))

plot(error_df_ridge$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RIDGE", ylab = "Errors", ylim = c(0.7,1))
lines(error_df_ridge$adj_r_sq,  col = 'green' )
lines(error_df_ridge$cv_r_sq,  col = 'blue')

#conc data error and plot                        
error_conc_ridge <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_conc_ridge <- add_row(error_conc_ridge, r_sq = max(rstats1(ridge_conc_mod1)$R2), adj_r_sq = max(rstats1(ridge_conc_mod1)$adjR2), cv_r_sq = mean(cv_ridge_conc1$resample$Rsquared))
error_conc_ridge <- add_row(error_conc_ridge, r_sq = max(rstats1(ridge_conc_mod2)$R2), adj_r_sq = max(rstats1(ridge_conc_mod2)$adjR2), cv_r_sq = mean(cv_ridge_conc2$resample$Rsquared))
error_conc_ridge <- add_row(error_conc_ridge, r_sq = max(rstats1(ridge_conc_mod3)$R2), adj_r_sq = max(rstats1(ridge_conc_mod3)$adjR2), cv_r_sq = mean(cv_ridge_conc3$resample$Rsquared))  
error_conc_ridge <- add_row(error_conc_ridge, r_sq = max(rstats1(ridge_conc_mod4)$R2), adj_r_sq = max(rstats1(ridge_conc_mod4)$adjR2), cv_r_sq = mean(cv_ridge_conc4$resample$Rsquared))
error_conc_ridge <- add_row(error_conc_ridge, r_sq = max(rstats1(ridge_conc_mod5)$R2), adj_r_sq = max(rstats1(ridge_conc_mod5)$adjR2), cv_r_sq = mean(cv_ridge_conc5$resample$Rsquared))
error_conc_ridge <- add_row(error_conc_ridge, r_sq = max(rstats1(ridge_conc_mod6)$R2), adj_r_sq = max(rstats1(ridge_conc_mod6)$adjR2), cv_r_sq = mean(cv_ridge_conc6$resample$Rsquared))
error_conc_ridge <- add_row(error_conc_ridge, r_sq = max(rstats1(ridge_conc_mod7)$R2), adj_r_sq = max(rstats1(ridge_conc_mod7)$adjR2), cv_r_sq = mean(cv_ridge_conc7$resample$Rsquared))


plot(error_conc_ridge$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RIDGE", ylab = "Errors", ylim = c(0,1))
lines(error_conc_ridge$adj_r_sq,  col = 'green' )
lines(error_conc_ridge$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)



### LASSO REGRESSION

#auto
x <- (auto_df$weight)
x <- cbind(auto_df$model.year, x)
lasso_mod1 <- lars(x, auto_df$mpg, type = 'lasso')

x <- (auto_df$weight)
x <- cbind(auto_df$model.year, x)
x <- cbind(auto_df$horsepower, x)
lasso_mod2 <- lars(x, auto_df$mpg, type = 'lasso')


x <- (auto_df$weight)
x <- cbind(auto_df$model.year, x)
x <- cbind(auto_df$horsepower, x)
x <- cbind(auto_df$origin, x)
lasso_mod3 <- lars(x, auto_df$mpg, type = 'lasso')


x <- (auto_df$weight)
x <- cbind(auto_df$model.year, x)
x <- cbind(auto_df$horsepower, x)
x <- cbind(auto_df$origin, x)
x <- cbind(auto_df$acceleration, x)
lasso_mod4 <- lars(x, auto_df$mpg, type = 'lasso')

x <- (auto_df$weight)
x <- cbind(auto_df$model.year, x)
x <- cbind(auto_df$horsepower, x)
x <- cbind(auto_df$origin, x)
x <- cbind(auto_df$acceleration, x)
x <- cbind(auto_df$cylinders, x)
lasso_mod5 <- lars(x, auto_df$mpg, type = 'lasso')


x <- (auto_df$weight)
x <- cbind(auto_df$model.year, x)
x <- cbind(auto_df$horsepower, x)
x <- cbind(auto_df$origin, x)
x <- cbind(auto_df$acceleration, x)
x <- cbind(auto_df$cylinders, x)
x <- cbind(auto_df$displacement, x)
lasso_mod6 <- lars(x, auto_df$mpg, type = 'lasso')

cv_lasso_model1 <- train(mpg ~ weight + model.year, data = auto_df, trControl = train_control, method = "lasso")
cv_lasso_model2 <- train(mpg ~ weight + model.year + horsepower, data = auto_df, trControl = train_control, method = "lasso")
cv_lasso_model3 <- train(mpg ~ weight + model.year + horsepower + origin, data = auto_df, trControl = train_control, method = "lasso")
cv_lasso_model4 <- train(mpg ~ weight + model.year + horsepower + origin + acceleration, data = auto_df, trControl = train_control, method = "lasso")
cv_lasso_model5 <- train(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders, data = auto_df, trControl = train_control, method = "lasso")
cv_lasso_model6 <- train(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders + displacement, data = auto_df, trControl = train_control, method = "lasso")

error_df_lasso <- data.frame("r_sq" = double(0), "cv_r_sq" = double(0))
error_df_lasso <- add_row(error_df_lasso, r_sq = max(lasso_mod1$R2), cv_r_sq = mean(cv_lasso_model1$resample$Rsquared))
error_df_lasso <- add_row(error_df_lasso, r_sq = max(lasso_mod2$R2), cv_r_sq = mean(cv_lasso_model2$resample$Rsquared))
error_df_lasso <- add_row(error_df_lasso, r_sq = max(lasso_mod3$R2), cv_r_sq = mean(cv_lasso_model3$resample$Rsquared))
error_df_lasso <- add_row(error_df_lasso, r_sq = max(lasso_mod4$R2), cv_r_sq = mean(cv_lasso_model4$resample$Rsquared))
error_df_lasso <- add_row(error_df_lasso, r_sq = max(lasso_mod5$R2), cv_r_sq = mean(cv_lasso_model5$resample$Rsquared))
error_df_lasso <- add_row(error_df_lasso, r_sq = max(lasso_mod6$R2), cv_r_sq = mean(cv_lasso_model6$resample$Rsquared))

#plot of r square, adjusted r square 
plot(error_df_lasso$r_sq, type = 'l', col = 'red', main = "ERROR PLOT", ylab = "Errors", ylim = c(0.3,1))
lines(error_df_lasso$cv_r_sq,  col = 'green' )
legend(5,0.8, legend = c("R Squared","CV R Squared"), 
       col = c("red","green"), lty = 1:2, cex = 0.8)




#conc data
x <- (conc_df$Cement)
x <- cbind(conc_df$Plasticizer, x)
lasso_conc_mod1 <- lars(x, conc_df$CCStrength, type = 'lasso')

x <- (conc_df$Cement)
x <- cbind(conc_df$Plasticizer, x)
x <- cbind(conc_df$Age, x)
lasso_conc_mod2 <- lars(x, conc_df$CCStrength, type = 'lasso')


x <- (conc_df$Cement)
x <- cbind(conc_df$Plasticizer, x)
x <- cbind(conc_df$Age, x)
x <- cbind(conc_df$Slag, x)
lasso_conc_mod3 <- lars(x, conc_df$CCStrength, type = 'lasso')


x <- (conc_df$Cement)
x <- cbind(conc_df$Plasticizer, x)
x <- cbind(conc_df$Age, x)
x <- cbind(conc_df$Slag, x)
x <- cbind(conc_df$Water, x)
lasso_conc_mod4 <- lars(x, conc_df$CCStrength, type = 'lasso')

x <- (conc_df$Cement)
x <- cbind(conc_df$Plasticizer, x)
x <- cbind(conc_df$Age, x)
x <- cbind(conc_df$Slag, x)
x <- cbind(conc_df$Water, x)
x <- cbind(conc_df$Flyash, x)
lasso_conc_mod5 <- lars(x, conc_df$CCStrength, type = 'lasso')

x <- (conc_df$Cement)
x <- cbind(conc_df$Plasticizer, x)
x <- cbind(conc_df$Age, x)
x <- cbind(conc_df$Slag, x)
x <- cbind(conc_df$Water, x)
x <- cbind(conc_df$Flyash, x)
x <- cbind(conc_df$FAggregate, x)
lasso_conc_mod6 <- lars(x, conc_df$CCStrength, type = 'lasso')

x <- (conc_df$Cement)
x <- cbind(conc_df$Plasticizer, x)
x <- cbind(conc_df$Age, x)
x <- cbind(conc_df$Slag, x)
x <- cbind(conc_df$Water, x)
x <- cbind(conc_df$Flyash, x)
x <- cbind(conc_df$FAggregate, x)
x <- cbind(conc_df$CAggregate, x)
lasso_conc_mod7 <- lars(x, conc_df$CCStrength, type = 'lasso')

cv_conc_lasso1 <- train(CCStrength ~ Cement + Plasticizer, data = conc_df, trControl = train_control, method = "lasso")
cv_conc_lasso2 <- train(CCStrength ~ Cement + Plasticizer + Age, data = conc_df, trControl = train_control, method = "lasso")
cv_conc_lasso3 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag, data = conc_df, trControl = train_control, method = "lasso")
cv_conc_lasso4 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag + Water, data = conc_df, trControl = train_control, method = "lasso")
cv_conc_lasso5 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash, data = conc_df, trControl = train_control, method = "lasso")
cv_conc_lasso6 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash + FAggregate, data = conc_df, trControl = train_control, method = "lasso")
cv_conc_lasso7 <- train(CCStrength ~ Cement + Plasticizer + Age + Slag + Water + Flyash + FAggregate + CAggregate, data = conc_df, trControl = train_control, method = "lasso")

error_lasso_conc <- data.frame("r_sq" = double(0), "cv_r_sq" = double(0))
error_lasso_conc <- add_row(error_lasso_conc, r_sq = max(lasso_conc_mod1$R2), cv_r_sq = mean(cv_conc_lasso1$resample$Rsquared))
error_lasso_conc <- add_row(error_lasso_conc, r_sq = max(lasso_conc_mod2$R2), cv_r_sq = mean(cv_conc_lasso2$resample$Rsquared))
error_lasso_conc <- add_row(error_lasso_conc, r_sq = max(lasso_conc_mod3$R2), cv_r_sq = mean(cv_conc_lasso3$resample$Rsquared))
error_lasso_conc <- add_row(error_lasso_conc, r_sq = max(lasso_conc_mod4$R2), cv_r_sq = mean(cv_conc_lasso4$resample$Rsquared))
error_lasso_conc <- add_row(error_lasso_conc, r_sq = max(lasso_conc_mod5$R2), cv_r_sq = mean(cv_conc_lasso5$resample$Rsquared))
error_lasso_conc <- add_row(error_lasso_conc, r_sq = max(lasso_conc_mod6$R2), cv_r_sq = mean(cv_conc_lasso6$resample$Rsquared))
error_lasso_conc <- add_row(error_lasso_conc, r_sq = max(lasso_conc_mod7$R2), cv_r_sq = mean(cv_conc_lasso7$resample$Rsquared))

plot(error_lasso_conc$r_sq, type = 'l', col = 'red', main = "ERROR PLOT LASSO", ylab = "Errors",xlab = "num variables", ylim = c(0.3,1))
lines(error_lasso_conc$cv_r_sq,  col = 'green' )
legend(5,0.8, legend = c("R Squared","CV R Squared"), 
       col = c("red","green"), lty = 1:2, cex = 0.8)

#### QUAD REGRESSION

#transferring the current data into new data frame
quad_data <- auto_df 

##adding square of each of the predictors to the data frame for quad regression
quad_data$weight_square <- quad_data$weight^2
quad_data$model.year_square <- quad_data$model.year^2
quad_data$horsepower_square <- quad_data$horsepower^2
quad_data$origin_square <- quad_data$origin^2
quad_data$acceleration_square <- quad_data$acceleration^2
quad_data$cylinders_square <- quad_data$cylinders^2
quad_data$displacement_square <- quad_data$displacement^2


#regression
quad_mod1 <- lm(mpg ~ weight + weight_square, data = quad_data)
quad_mod2 <- lm(mpg ~ weight + weight_square + model.year + model.year_square, data = quad_data)
quad_mod3 <- lm(mpg ~ weight + weight_square + model.year + model.year_square + horsepower + horsepower_square, data = quad_data)
quad_mod4 <- lm(mpg ~ weight + weight_square + model.year + model.year_square + horsepower + horsepower_square + origin + origin_square, data = quad_data)
quad_mod5 <- lm(mpg ~ weight + weight_square + model.year + model.year_square + horsepower + horsepower_square + origin + origin_square + acceleration + acceleration_square, data = quad_data)
quad_mod6 <- lm(mpg ~ weight + weight_square + model.year + model.year_square + horsepower + horsepower_square + origin + origin_square + acceleration + acceleration_square + cylinders + cylinders_square, data = quad_data)
quad_mod7 <- lm(mpg ~ weight + weight_square + model.year + model.year_square + horsepower + horsepower_square + origin + origin_square + acceleration + acceleration_square + cylinders + cylinders_square + displacement + displacement_square, data = quad_data)

#cross validation
cv_quad_mod1 <- train(mpg ~ weight + weight_square, data = quad_data, trControl = train_control, method = "lm")
cv_quad_mod2 <- train(mpg ~ weight + weight_square + model.year + model.year_square, data = quad_data, trControl = train_control, method = "lm")
cv_quad_mod3 <- train(mpg ~ weight + weight_square + model.year + model.year_square + horsepower + horsepower_square, data = quad_data, trControl = train_control, method = "lm")
cv_quad_mod4 <- train(mpg ~ weight + weight_square + model.year + model.year_square + horsepower + horsepower_square + origin + origin_square, data = quad_data, trControl = train_control, method = "lm")
cv_quad_mod5 <- train(mpg ~ weight + weight_square + model.year + model.year_square + horsepower + horsepower_square + origin + origin_square + acceleration + acceleration_square, data = quad_data, trControl = train_control, method = "lm")
cv_quad_mod6 <- train(mpg ~ weight + weight_square + model.year + model.year_square + horsepower + horsepower_square + origin + origin_square + acceleration + acceleration_square + cylinders + cylinders_square, data = quad_data, trControl = train_control, method = "lm")
cv_quad_mod7 <- train(mpg ~ weight + weight_square + model.year + model.year_square + horsepower + horsepower_square + origin + origin_square + acceleration + acceleration_square + cylinders + cylinders_square + displacement + displacement_square, data = quad_data, trControl = train_control, method = "lm")

#creating a new dataframe to store rsquare adjusted_rsquard and cv rsquared
error_df_quad <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_df_quad <- add_row(error_df_lm, r_sq = summary(quad_mod1)$r.squared, adj_r_sq = summary(quad_mod1)$adj.r.squared, cv_r_sq = mean(cv_quad_mod1$resample$Rsquared))
error_df_quad <- add_row(error_df_lm, r_sq = summary(quad_mod2)$r.squared, adj_r_sq = summary(quad_mod2)$adj.r.squared, cv_r_sq = mean(cv_quad_mod2$resample$Rsquared))
error_df_quad <- add_row(error_df_lm, r_sq = summary(quad_mod3)$r.squared, adj_r_sq = summary(quad_mod3)$adj.r.squared, cv_r_sq = mean(cv_quad_mod3$resample$Rsquared))
error_df_quad <- add_row(error_df_lm, r_sq = summary(quad_mod4)$r.squared, adj_r_sq = summary(quad_mod4)$adj.r.squared, cv_r_sq = mean(cv_quad_mod4$resample$Rsquared))
error_df_quad <- add_row(error_df_lm, r_sq = summary(quad_mod5)$r.squared, adj_r_sq = summary(quad_mod5)$adj.r.squared, cv_r_sq = mean(cv_quad_mod5$resample$Rsquared))
error_df_quad <- add_row(error_df_lm, r_sq = summary(quad_mod6)$r.squared, adj_r_sq = summary(quad_mod6)$adj.r.squared, cv_r_sq = mean(cv_quad_mod6$resample$Rsquared))
error_df_quad <- add_row(error_df_lm, r_sq = summary(quad_mod7)$r.squared, adj_r_sq = summary(quad_mod7)$adj.r.squared, cv_r_sq = mean(cv_quad_mod7$resample$Rsquared))

#plot of r square, adjusted r square and r square cross-validation
plot(error_df_quad$r_sq, type = 'l', col = 'red', main = "ERROR PLOT QUAD", ylab = "Errors", ylim = c(0.7,1))
lines(error_df_quad$adj_r_sq,  col = 'green' )
lines(error_df_quad$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)

#conc data
#transferring the current data into new data frame
data2_quad <- conc_df 

#adding squares of each term for quad
data2_quad$Cement_square <- data2_quad$Cement^2
data2_quad$Plasticizer_square <- data2_quad$Plasticizer^2
data2_quad$Age_square <- data2_quad$Age^2
data2_quad$Slag_square <- data2_quad$Slag^2
data2_quad$Water_square <- data2_quad$Water^2
data2_quad$Flyash_square <- data2_quad$Flyash^2
data2_quad$FAggregate_square <- data2_quad$FAggregate^2
data2_quad$CAggregate_square <- data2_quad$CAggregate^2



data2_quad_mod1 <- lm(CCStrength ~ Cement + Cement_square, data = data2_quad)
data2_quad_mod2 <- lm(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square, data = data2_quad)
data2_quad_mod3 <- lm(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square + Age + Age_square, data = data2_quad)
data2_quad_mod4 <- lm(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square + Age + Age_square + Slag + Slag_square, data = data2_quad)
data2_quad_mod5 <- lm(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square + Age + Age_square + Slag + Slag_square + Water + Water_square, data = data2_quad)
data2_quad_mod6 <- lm(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square + Age + Age_square + Slag + Slag_square + Water + Water_square + Flyash + Flyash_square, data = data2_quad)
data2_quad_mod7 <- lm(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square + Age + Age_square + Slag + Slag_square + Water + Water_square + Flyash + Flyash_square + FAggregate + FAggregate_square, data = data2_quad)
data2_quad_mod8 <- lm(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square + Age + Age_square + Slag + Slag_square + Water + Water_square + Flyash + Flyash_square + FAggregate + FAggregate_square + CAggregate + CAggregate_square, data = data2_quad)

cv_data2_quad1 <- train(CCStrength ~ Cement + Cement_square, data = data2_quad, trControl = train_control, method = "lm")
cv_data2_quad2 <- train(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square, data = data2_quad, trControl = train_control, method = "lm")
cv_data2_quad3 <- train(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square + Age + Age_square, data = data2_quad, trControl = train_control, method = "lm")
cv_data2_quad4 <- train(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square + Age + Age_square + Slag + Slag_square, data = data2_quad, trControl = train_control, method = "lm")
cv_data2_quad5 <- train(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square + Age + Age_square + Slag + Slag_square + Water + Water_square, data = data2_quad, trControl = train_control, method = "lm")
cv_data2_quad6 <- train(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square + Age + Age_square + Slag + Slag_square + Water + Water_square + Flyash + Flyash_square, data = data2_quad, trControl = train_control, method = "lm")
cv_data2_quad7 <- train(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square + Age + Age_square + Slag + Slag_square + Water + Water_square + Flyash + Flyash_square + FAggregate + FAggregate_square, data = data2_quad, trControl = train_control, method = "lm")
cv_data2_quad8 <- train(CCStrength ~ Cement + Cement_square + Plasticizer + Plasticizer_square + Age + Age_square + Slag + Slag_square + Water + Water_square + Flyash + Flyash_square + FAggregate + FAggregate_square + CAggregate + CAggregate_square, data = data2_quad, trControl = train_control, method = "lm")



error_data2_quad <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data2_quad <- add_row(error_data2_quad, r_sq = summary(data2_quad_mod1)$r.squared, adj_r_sq = summary(data2_quad_mod1)$adj.r.squared, cv_r_sq = mean(cv_data2_quad1$resample$Rsquared))
error_data2_quad <- add_row(error_data2_quad, r_sq = summary(data2_quad_mod2)$r.squared, adj_r_sq = summary(data2_quad_mod2)$adj.r.squared, cv_r_sq = mean(cv_data2_quad2$resample$Rsquared))
error_data2_quad <- add_row(error_data2_quad, r_sq = summary(data2_quad_mod3)$r.squared, adj_r_sq = summary(data2_quad_mod3)$adj.r.squared, cv_r_sq = mean(cv_data2_quad3$resample$Rsquared))
error_data2_quad <- add_row(error_data2_quad, r_sq = summary(data2_quad_mod4)$r.squared, adj_r_sq = summary(data2_quad_mod4)$adj.r.squared, cv_r_sq = mean(cv_data2_quad4$resample$Rsquared))
error_data2_quad <- add_row(error_data2_quad, r_sq = summary(data2_quad_mod5)$r.squared, adj_r_sq = summary(data2_quad_mod5)$adj.r.squared, cv_r_sq = mean(cv_data2_quad5$resample$Rsquared))
error_data2_quad <- add_row(error_data2_quad, r_sq = summary(data2_quad_mod6)$r.squared, adj_r_sq = summary(data2_quad_mod6)$adj.r.squared, cv_r_sq = mean(cv_data2_quad6$resample$Rsquared))
error_data2_quad <- add_row(error_data2_quad, r_sq = summary(data2_quad_mod7)$r.squared, adj_r_sq = summary(data2_quad_mod7)$adj.r.squared, cv_r_sq = mean(cv_data2_quad7$resample$Rsquared))
error_data2_quad <- add_row(error_data2_quad, r_sq = summary(data2_quad_mod8)$r.squared, adj_r_sq = summary(data2_quad_mod8)$adj.r.squared, cv_r_sq = mean(cv_data2_quad8$resample$Rsquared))



plot(error_data2_quad$r_sq, type = 'l', col = 'red', main = "ERROR PLOT QUAD", ylab = "Errors", xlab = "num variable", ylim = c(0,1))
lines(error_data2_quad$adj_r_sq,  col = 'green' )
lines(error_data2_quad$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)





## RESPONSE SURFACE

#creating a new dataframe from auto df for RESPONSE SURFACE REGRESSION
response_df <- auto_df

rspns_mod1 <- rsm(mpg ~ SO(weight, model.year), data = response_df)
rspns_mod2 <- rsm(mpg ~ SO(weight, model.year, horsepower), data = response_df)
rspns_mod3 <- rsm(mpg ~ SO(weight, model.year, horsepower, origin), data = response_df)
rspns_mod4 <- rsm(mpg ~ SO(weight, model.year, horsepower, origin, acceleration), data = response_df)
rspns_mod5 <- rsm(mpg ~ SO(weight, model.year, horsepower, origin, acceleration, cylinders), data = response_df)
rspns_mod6 <- rsm(mpg ~ SO(weight, model.year, horsepower, origin, acceleration, cylinders, displacement), data = response_df)


error_df_rspns <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0))

error_df_rspns <- add_row(error_df_rspns, r_sq = summary(rspns_mod1)$r.squared, adj_r_sq = summary(rspns_mod1)$adj.r.squared)
error_df_rspns <- add_row(error_df_rspns, r_sq = summary(rspns_mod2)$r.squared, adj_r_sq = summary(rspns_mod2)$adj.r.squared)
error_df_rspns <- add_row(error_df_rspns, r_sq = summary(rspns_mod3)$r.squared, adj_r_sq = summary(rspns_mod3)$adj.r.squared)
error_df_rspns <- add_row(error_df_rspns, r_sq = summary(rspns_mod4)$r.squared, adj_r_sq = summary(rspns_mod4)$adj.r.squared)
error_df_rspns <- add_row(error_df_rspns, r_sq = summary(rspns_mod5)$r.squared, adj_r_sq = summary(rspns_mod5)$adj.r.squared)
error_df_rspns <- add_row(error_df_rspns, r_sq = summary(rspns_mod6)$r.squared, adj_r_sq = summary(rspns_mod6)$adj.r.squared)

#plot of r square, adjusted r square
plot(error_df_rspns$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RESPONSE SURFACE", ylab = "Errors", ylim = c(0.3,1))
lines(error_df_rspns$adj_r_sq,  col = 'green' )
legend(5,0.8, legend = c("R Squared","Adj R Squared"), 
       col = c("red","green"), lty = 1:2, cex = 0.8)


#creating a new dataframe from auto df for RESPONSE SURFACE REGRESSION
data2_rspns <- conc_df

data2_rspns_mod1 <- rsm(CCStrength ~ SO(Cement, Plasticizer), data = data2_rspns)
data2_rspns_mod2 <- rsm(CCStrength ~ SO(Cement, Plasticizer, Age), data = data2_rspns)
data2_rspns_mod3 <- rsm(CCStrength ~ SO(Cement, Plasticizer, Age, Slag), data = data2_rspns)
data2_rspns_mod4 <- rsm(CCStrength ~ SO(Cement, Plasticizer, Age, Slag, Water), data = data2_rspns)
data2_rspns_mod5 <- rsm(CCStrength ~ SO(Cement, Plasticizer, Age, Slag, Water, Flyash), data = data2_rspns)
data2_rspns_mod6 <- rsm(CCStrength ~ SO(Cement, Plasticizer, Age, Slag, Water, Flyash, FAggregate), data = data2_rspns)
data2_rspns_mod7 <- rsm(CCStrength ~ SO(Cement, Plasticizer, Age, Slag, Water, Flyash, FAggregate, CAggregate), data = data2_rspns)


data2_error_rspns <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0))

data2_error_rspns <- add_row(data2_error_rspns, r_sq = summary(data2_rspns_mod1)$r.squared, adj_r_sq = summary(data2_rspns_mod1)$adj.r.squared)
data2_error_rspns <- add_row(data2_error_rspns, r_sq = summary(data2_rspns_mod2)$r.squared, adj_r_sq = summary(data2_rspns_mod2)$adj.r.squared)
data2_error_rspns <- add_row(data2_error_rspns, r_sq = summary(data2_rspns_mod3)$r.squared, adj_r_sq = summary(data2_rspns_mod3)$adj.r.squared)
data2_error_rspns <- add_row(data2_error_rspns, r_sq = summary(data2_rspns_mod4)$r.squared, adj_r_sq = summary(data2_rspns_mod4)$adj.r.squared)
data2_error_rspns <- add_row(data2_error_rspns, r_sq = summary(data2_rspns_mod5)$r.squared, adj_r_sq = summary(data2_rspns_mod5)$adj.r.squared)
data2_error_rspns <- add_row(data2_error_rspns, r_sq = summary(data2_rspns_mod6)$r.squared, adj_r_sq = summary(data2_rspns_mod6)$adj.r.squared)

#plot of r square, adjusted r square
plot(data2_error_rspns$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RESPONSE SURFACE", ylab = "Errors",xlab = "num variable", ylim = c(0.3,1))
lines(data2_error_rspns$adj_r_sq,  col = 'green' )
legend(5,0.8, legend = c("R Squared","Adj R Squared"), 
       col = c("red","green"), lty = 1:2, cex = 0.8)


### DATASET3  YACHT DATA

df <- read.csv('/home/mr_malviya/Desktop/yacht.csv')
data3 <- df

#replacing the missing values with mean
data3$Longitudnal_position[is.na(data3$Longitudnal_position)] <- mean(data3$Longitudnal_position, na.rm = TRUE)
data3$Prismatic_coef[is.na(data3$Prismatic_coef)] <- mean(data3$Prismatic_coef, na.rm = TRUE)
data3$Length_displacement[is.na(data3$Length_displacement)] <- mean(data3$Length_displacement, na.rm = TRUE)
data3$Beam_draught[is.na(data3$Beam_draught)] <- mean(data3$Beam_draught, na.rm = TRUE)
data3$length_beam[is.na(data3$length_beam)] <- mean(data3$length_beam, na.rm = TRUE)
data3$froude_number[is.na(data3$froude_number)] <- mean(data3$froude_number, na.rm = TRUE)
data3$residuary_resistance[is.na(data3$residuary_resistance)] <- mean(data3$residuary_resistance, na.rm = TRUE)

#getting the features using forward selection
fwd_model <- lm(residuary_resistance ~ 1,  data = data3) 
step(fwd_model, direction = "forward", scope = formula(residuary_resistance ~ Longitudnal_position + Prismatic_coef + Length_displacement + Beam_draught + length_beam + froude_number))
summary(fwd_model_auto)
# formula obtained residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught + Prismatic_coef

# LINEAR REGRESSION
data3_lin1 <- lm(residuary_resistance ~ Length_displacement, data = data3)
data3_lin2 <- lm(residuary_resistance ~ Length_displacement + froude_number, data = data3)
data3_lin3 <- lm(residuary_resistance ~ Length_displacement + froude_number + length_beam, data = data3)
data3_lin4 <- lm(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught, data = data3)
data3_lin5 <- lm(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught + Prismatic_coef, data = data3)
data3_lin6 <- lm(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught + Prismatic_coef + Longitudnal_position, data = data3)


cv_data3_lin1 <- train(residuary_resistance ~ Length_displacement, data = data3, trControl = train_control, method = "lm")
cv_data3_lin2 <- train(residuary_resistance ~ Length_displacement + froude_number, data = data3, trControl = train_control, method = "lm")
cv_data3_lin3 <- train(residuary_resistance ~ Length_displacement + froude_number + length_beam, data = data3, trControl = train_control, method = "lm")
cv_data3_lin4 <- train(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught, data = data3, trControl = train_control, method = "lm")
cv_data3_lin5 <- train(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught + Prismatic_coef, data = data3, trControl = train_control, method = "lm")
cv_data3_lin6 <- train(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught + Prismatic_coef + Longitudnal_position, data = data3, trControl = train_control, method = "lm")


error_data3_lin <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data3_lin <- add_row(error_data3_lin, r_sq = summary(data3_lin1)$r.squared, adj_r_sq = summary(data3_lin1)$adj.r.squared, cv_r_sq = mean(cv_data3_lin1$resample$Rsquared))
error_data3_lin <- add_row(error_data3_lin, r_sq = summary(data3_lin2)$r.squared, adj_r_sq = summary(data3_lin2)$adj.r.squared, cv_r_sq = mean(cv_data3_lin2$resample$Rsquared))
error_data3_lin <- add_row(error_data3_lin, r_sq = summary(data3_lin3)$r.squared, adj_r_sq = summary(data3_lin3)$adj.r.squared, cv_r_sq = mean(cv_data3_lin3$resample$Rsquared))
error_data3_lin <- add_row(error_data3_lin, r_sq = summary(data3_lin4)$r.squared, adj_r_sq = summary(data3_lin4)$adj.r.squared, cv_r_sq = mean(cv_data3_lin4$resample$Rsquared))
error_data3_lin <- add_row(error_data3_lin, r_sq = summary(data3_lin5)$r.squared, adj_r_sq = summary(data3_lin5)$adj.r.squared, cv_r_sq = mean(cv_data3_lin5$resample$Rsquared))
error_data3_lin <- add_row(error_data3_lin, r_sq = summary(data3_lin6)$r.squared, adj_r_sq = summary(data3_lin6)$adj.r.squared, cv_r_sq = mean(cv_data3_lin6$resample$Rsquared))



#plot of r square, adjusted r square and r square cross-validation
plot(error_data3_lin$r_sq, type = 'l', col = 'red', main = "ERROR PLOT", ylab = "Errors", ylim = c(0.7,1))
lines(error_data3_lin$adj_r_sq,  col = 'green' )
lines(error_data3_lin$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


## RIDGE REGRESSION

data3_ridge1 <- lmridge(residuary_resistance ~ Length_displacement + froude_number, data3, K = c(0.1, 0.001))
data3_ridge2 <- lmridge(residuary_resistance ~ Length_displacement + froude_number + length_beam, data3, K = c(0.1, 0.001))
data3_ridge3 <- lmridge(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught, data3, K = c(0.1, 0.001))
data3_ridge4 <- lmridge(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught + Prismatic_coef , data3, K = c(0.1, 0.001))
data3_ridge5 <- lmridge(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught + Prismatic_coef + Longitudnal_position, data3, K = c(0.1, 0.001))


cv_data3_ridge1 <- train(residuary_resistance ~ Length_displacement + froude_number, data = data3, trControl = train_control, method = "ridge")
cv_data3_ridge2 <- train(residuary_resistance ~ Length_displacement + froude_number + length_beam, data = data3, trControl = train_control, method = "ridge")
cv_data3_ridge3 <- train(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught, data = data3, trControl = train_control, method = "ridge")
cv_data3_ridge4 <- train(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught + Prismatic_coef, data = data3, trControl = train_control, method = "ridge")
cv_data3_ridge5 <- train(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught + Prismatic_coef + Longitudnal_position, data = data3, trControl = train_control, method = "ridge")

error_data3_ridge <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data3_ridge <- add_row(error_data3_ridge, r_sq = max(rstats1(data3_ridge1)$R2), adj_r_sq = max(rstats1(data3_ridge1)$adjR2), cv_r_sq = mean(cv_data3_ridge1$resample$Rsquared))
error_data3_ridge <- add_row(error_data3_ridge, r_sq = max(rstats1(data3_ridge2)$R2), adj_r_sq = max(rstats1(data3_ridge2)$adjR2), cv_r_sq = mean(cv_data3_ridge2$resample$Rsquared))
error_data3_ridge <- add_row(error_data3_ridge, r_sq = max(rstats1(data3_ridge3)$R2), adj_r_sq = max(rstats1(data3_ridge3)$adjR2), cv_r_sq = mean(cv_data3_ridge3$resample$Rsquared))  
error_data3_ridge <- add_row(error_data3_ridge, r_sq = max(rstats1(data3_ridge4)$R2), adj_r_sq = max(rstats1(data3_ridge4)$adjR2), cv_r_sq = mean(cv_data3_ridge4$resample$Rsquared))
error_data3_ridge <- add_row(error_data3_ridge, r_sq = max(rstats1(data3_ridge5)$R2), adj_r_sq = max(rstats1(data3_ridge5)$adjR2), cv_r_sq = mean(cv_data3_ridge5$resample$Rsquared))

plot(error_data3_ridge$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RIDGE", ylab = "Errors",xlab = "num variable", ylim = c(0,1))
lines(error_data3_ridge$adj_r_sq,  col = 'green' )
lines(error_data3_ridge$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


##  LASSO REGRESSION
x <- (data3$Length_displacement)
x <- cbind(data3$froude_number, x)
data3_lasso1 <- lars(x, data3$residuary_resistance, type = 'lasso')

x <- (data3$Length_displacement)
x <- cbind(data3$froude_number, x)
x <- cbind(data3$length_beam, x)
data3_lasso2 <- lars(x, data3$residuary_resistance, type = 'lasso')


x <- (data3$Length_displacement)
x <- cbind(data3$froude_number, x)
x <- cbind(data3$length_beam, x)
x <- cbind(data3$Beam_draught, x)
data3_lasso3 <- lars(x, data3$residuary_resistance, type = 'lasso')


x <- (data3$Length_displacement)
x <- cbind(data3$froude_number, x)
x <- cbind(data3$length_beam, x)
x <- cbind(data3$Beam_draught, x)
x <- cbind(data3$Prismatic_coef, x)
data3_lasso4 <- lars(x, data3$residuary_resistance, type = 'lasso')

x <- (data3$Length_displacement)
x <- cbind(data3$froude_number, x)
x <- cbind(data3$length_beam, x)
x <- cbind(data3$Beam_draught, x)
x <- cbind(data3$Prismatic_coef, x)
x <- cbind(data3$Longitudnal_position, x)
data3_lasso5 <- lars(x, data3$residuary_resistance, type = 'lasso')


x <- (data3$Length_displacement)
x <- cbind(data3$froude_number, x)
x <- cbind(data3$length_beam, x)
x <- cbind(data3$Beam_draught, x)
x <- cbind(data3$Prismatic_coef, x)
x <- cbind(data3$Longitudnal_position, x)
x <- cbind(data3$displacement, x)
data3_lasso6 <- lars(x, data3$residuary_resistance, type = 'lasso')

cv_data3_lasso1 <- train(residuary_resistance ~ Length_displacement + froude_number, data = data3, trControl = train_control, method = "lasso")
cv_data3_lasso2 <- train(residuary_resistance ~ Length_displacement + froude_number + length_beam, data = data3, trControl = train_control, method = "lasso")
cv_data3_lasso3 <- train(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught, data = data3, trControl = train_control, method = "lasso")
cv_data3_lasso4 <- train(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught + Prismatic_coef, data = data3, trControl = train_control, method = "lasso")
cv_data3_lasso5 <- train(residuary_resistance ~ Length_displacement + froude_number + length_beam + Beam_draught + Prismatic_coef + Longitudnal_position, data = data3, trControl = train_control, method = "lasso")

error_data3_lasso <- data.frame("r_sq" = double(0), "cv_r_sq" = double(0))
error_data3_lasso <- add_row(error_data3_lasso, r_sq = max(data3_lasso1$R2), cv_r_sq = mean(cv_data3_lasso1$resample$Rsquared))
error_data3_lasso <- add_row(error_data3_lasso, r_sq = max(data3_lasso2$R2), cv_r_sq = mean(cv_data3_lasso2$resample$Rsquared))
error_data3_lasso <- add_row(error_data3_lasso, r_sq = max(data3_lasso3$R2), cv_r_sq = mean(cv_data3_lasso3$resample$Rsquared))
error_data3_lasso <- add_row(error_data3_lasso, r_sq = max(data3_lasso4$R2), cv_r_sq = mean(cv_data3_lasso4$resample$Rsquared))
error_data3_lasso <- add_row(error_data3_lasso, r_sq = max(data3_lasso5$R2), cv_r_sq = mean(cv_data3_lasso5$resample$Rsquared))

#plot of r square, adjusted r square 
plot(error_data3_lasso$r_sq, type = 'l', col = 'red', main = "ERROR PLOT LASSO", ylab = "Errors",xlab = "num variable", ylim = c(0,1))
lines(error_data3_ridge$adj_r_sq,  col = 'green' )
lines(error_data3_ridge$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


## QUAD REGRESSION
##adding square of each of the predictors to the data frame for quad regression
data3$Length_displacement_square <- data3$Length_displacement^2
data3$froude_number_square <- data3$froude_number^2
data3$length_beam_square <- data3$length_beam^2
data3$Beam_draught_square <- data3$Beam_draught^2
data3$Prismatic_coef_square <- data3$Prismatic_coef^2
data3$Longitudnal_position_square <- data3$Longitudnal_position^2


#regression
data3_quad1 <- lm(residuary_resistance ~ Length_displacement + Length_displacement_square, data = data3)
data3_quad2 <- lm(residuary_resistance ~ Length_displacement + Length_displacement_square + froude_number + froude_number_square, data = data3)
data3_quad3 <- lm(residuary_resistance ~ Length_displacement + Length_displacement_square + froude_number + froude_number_square + length_beam + length_beam_square, data = data3)
data3_quad4 <- lm(residuary_resistance ~ Length_displacement + Length_displacement_square + froude_number + froude_number_square + length_beam + length_beam_square + Beam_draught + Beam_draught_square, data = data3)
data3_quad5 <- lm(residuary_resistance ~ Length_displacement + Length_displacement_square + froude_number + froude_number_square + length_beam + length_beam_square + Beam_draught + Beam_draught_square + Prismatic_coef + Prismatic_coef_square, data = data3)
data3_quad6 <- lm(residuary_resistance ~ Length_displacement + Length_displacement_square + froude_number + froude_number_square + length_beam + length_beam_square + Beam_draught + Beam_draught_square + Prismatic_coef + Prismatic_coef_square + Longitudnal_position + Longitudnal_position_square, data = data3)

#cross validation
cv_data3_quad1 <- train(residuary_resistance ~ Length_displacement + Length_displacement_square, data = data3, trControl = train_control, method = "lm")
cv_data3_quad2 <- train(residuary_resistance ~ Length_displacement + Length_displacement_square + froude_number + froude_number_square, data = data3, trControl = train_control, method = "lm")
cv_data3_quad3 <- train(residuary_resistance ~ Length_displacement + Length_displacement_square + froude_number + froude_number_square + length_beam + length_beam_square, data = data3, trControl = train_control, method = "lm")
cv_data3_quad4 <- train(residuary_resistance ~ Length_displacement + Length_displacement_square + froude_number + froude_number_square + length_beam + length_beam_square + Beam_draught + Beam_draught_square, data = data3, trControl = train_control, method = "lm")
cv_data3_quad5 <- train(residuary_resistance ~ Length_displacement + Length_displacement_square + froude_number + froude_number_square + length_beam + length_beam_square + Beam_draught + Beam_draught_square + Prismatic_coef + Prismatic_coef_square, data = data3, trControl = train_control, method = "lm")
cv_data3_quad6 <- train(residuary_resistance ~ Length_displacement + Length_displacement_square + froude_number + froude_number_square + length_beam + length_beam_square + Beam_draught + Beam_draught_square + Prismatic_coef + Prismatic_coef_square + Longitudnal_position + Longitudnal_position_square, data = data3, trControl = train_control, method = "lm")

#creating a new dataframe to store rsquare adjusted_rsquard and cv rsquared
error_data3_quad <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data3_quad <- add_row(error_data3_quad, r_sq = summary(data3_quad1)$r.squared, adj_r_sq = summary(data3_quad1)$adj.r.squared, cv_r_sq = mean(cv_data3_quad1$resample$Rsquared))
error_data3_quad <- add_row(error_data3_quad, r_sq = summary(data3_quad2)$r.squared, adj_r_sq = summary(data3_quad2)$adj.r.squared, cv_r_sq = mean(cv_data3_quad2$resample$Rsquared))
error_data3_quad <- add_row(error_data3_quad, r_sq = summary(data3_quad3)$r.squared, adj_r_sq = summary(data3_quad3)$adj.r.squared, cv_r_sq = mean(cv_data3_quad3$resample$Rsquared))
error_data3_quad <- add_row(error_data3_quad, r_sq = summary(data3_quad4)$r.squared, adj_r_sq = summary(data3_quad4)$adj.r.squared, cv_r_sq = mean(cv_data3_quad4$resample$Rsquared))
error_data3_quad <- add_row(error_data3_quad, r_sq = summary(data3_quad5)$r.squared, adj_r_sq = summary(data3_quad5)$adj.r.squared, cv_r_sq = mean(cv_data3_quad5$resample$Rsquared))
error_data3_quad <- add_row(error_data3_quad, r_sq = summary(data3_quad6)$r.squared, adj_r_sq = summary(data3_quad6)$adj.r.squared, cv_r_sq = mean(cv_data3_quad6$resample$Rsquared))

#plot of r square, adjusted r square and r square cross-validation
plot(error_data3_quad$r_sq, type = 'l', col = 'red', main = "ERROR PLOT QUAD", ylab = "Errors",xlab = "num variable", ylim = c(0.7,1))
lines(error_data3_quad$adj_r_sq,  col = 'green' )
lines(error_data3_quad$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


### RESPONSE SURFACE REGRESSION

data3_rspns1 <- rsm(residuary_resistance ~ SO(Length_displacement, froude_number), data = data3)
data3_rspns2 <- rsm(residuary_resistance ~ SO(Length_displacement, froude_number, length_beam), data = data3)
data3_rspns3 <- rsm(residuary_resistance ~ SO(Length_displacement, froude_number, length_beam, Beam_draught), data = data3)
data3_rspns4 <- rsm(residuary_resistance ~ SO(Length_displacement, froude_number, length_beam, Beam_draught, Prismatic_coef), data = data3)
data3_rspns5 <- rsm(residuary_resistance ~ SO(Length_displacement, froude_number, length_beam, Beam_draught, Prismatic_coef, Longitudnal_position), data = data3)


error_data3_rspns <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0))
error_data3_rspns <- add_row(error_data3_rspns, r_sq = summary(data3_rspns1)$r.squared, adj_r_sq = summary(data3_rspns1)$adj.r.squared)
error_data3_rspns <- add_row(error_data3_rspns, r_sq = summary(data3_rspns2)$r.squared, adj_r_sq = summary(data3_rspns2)$adj.r.squared)
error_data3_rspns <- add_row(error_data3_rspns, r_sq = summary(data3_rspns3)$r.squared, adj_r_sq = summary(data3_rspns3)$adj.r.squared)
error_data3_rspns <- add_row(error_data3_rspns, r_sq = summary(data3_rspns4)$r.squared, adj_r_sq = summary(data3_rspns4)$adj.r.squared)
error_data3_rspns <- add_row(error_data3_rspns, r_sq = summary(data3_rspns5)$r.squared, adj_r_sq = summary(data3_rspns5)$adj.r.squared)

#plot of r square, adjusted r square
plot(error_data3_rspns$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RESPONSE SURFACE", ylab = "Errors", ylim = c(0.3,1))
lines(error_data3_rspns$adj_r_sq,  col = 'green' )
legend(5,0.8, legend = c("R Squared","Adj R Squared"), 
       col = c("red","green"), lty = 1:2, cex = 0.8)




### REAL ESTATE DATA

data4 <- read.xls('/filepath')

data4$tran_date[is.na(data4$tran_date)] <- mean(data4$tran_date, na.rm = TRUE)
data4$house_age[is.na(data4$house_age)] <- mean(data4$house_age, na.rm = TRUE)
data4$distance[is.na(data4$distance)] <- mean(data4$distance, na.rm = TRUE)
data4$stores[is.na(data4$stores)] <- mean(data4$stores, na.rm = TRUE)
data4$latitude[is.na(data4$latitude)] <- mean(data4$latitude, na.rm = TRUE)
data4$longitude[is.na(data4$longitude)] <- mean(data4$longitude, na.rm = TRUE)
data4$price[is.na(data4$price)] <- mean(data4$price, na.rm = TRUE)

fwd_model_auto <- lm(price ~ 1,  data = data4) 
step(fwd_model_auto, direction = "forward", scope = formula(price ~ tran_date + house_age + distance + stores + latitude + longitude))
summary(fwd_model_auto)
# formula obtained price ~ distance + stores + house_age + latitude + tran_date

#auto price
data4_lin1 <- lm(price ~ distance, data = data4)
data4_lin2 <- lm(price ~ distance + stores, data = data4)
data4_lin3 <- lm(price ~ distance + stores + house_age, data = data4)
data4_lin4 <- lm(price ~ distance + stores + house_age + latitude, data = data4)
data4_lin5 <- lm(price ~ distance + stores + house_age + latitude + tran_date, data = data4)
data4_lin6 <- lm(price ~ distance + stores + house_age + latitude + tran_date + longitude, data = data4)


cv_data4_lin1 <- train(price ~ distance, data = data4, trControl = train_control, method = "lm")
cv_data4_lin2 <- train(price ~ distance + stores, data = data4, trControl = train_control, method = "lm")
cv_data4_lin3 <- train(price ~ distance + stores + house_age, data = data4, trControl = train_control, method = "lm")
cv_data4_lin4 <- train(price ~ distance + stores + house_age + latitude, data = data4, trControl = train_control, method = "lm")
cv_data4_lin5 <- train(price ~ distance + stores + house_age + latitude + tran_date, data = data4, trControl = train_control, method = "lm")
cv_data4_lin6 <- train(price ~ distance + stores + house_age + latitude + tran_date + longitude, data = data4, trControl = train_control, method = "lm")


error_data4_lin <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data4_lin <- add_row(error_data4_lin, r_sq = summary(data4_lin1)$r.squared, adj_r_sq = summary(data4_lin1)$adj.r.squared, cv_r_sq = mean(cv_data4_lin1$resample$Rsquared))
error_data4_lin <- add_row(error_data4_lin, r_sq = summary(data4_lin2)$r.squared, adj_r_sq = summary(data4_lin2)$adj.r.squared, cv_r_sq = mean(cv_data4_lin2$resample$Rsquared))
error_data4_lin <- add_row(error_data4_lin, r_sq = summary(data4_lin3)$r.squared, adj_r_sq = summary(data4_lin3)$adj.r.squared, cv_r_sq = mean(cv_data4_lin3$resample$Rsquared))
error_data4_lin <- add_row(error_data4_lin, r_sq = summary(data4_lin4)$r.squared, adj_r_sq = summary(data4_lin4)$adj.r.squared, cv_r_sq = mean(cv_data4_lin4$resample$Rsquared))
error_data4_lin <- add_row(error_data4_lin, r_sq = summary(data4_lin5)$r.squared, adj_r_sq = summary(data4_lin5)$adj.r.squared, cv_r_sq = mean(cv_data4_lin5$resample$Rsquared))
error_data4_lin <- add_row(error_data4_lin, r_sq = summary(data4_lin6)$r.squared, adj_r_sq = summary(data4_lin6)$adj.r.squared, cv_r_sq = mean(cv_data4_lin6$resample$Rsquared))



#plot of r square, adjusted r square and r square cross-validation
plot(error_data4_lin$r_sq, type = 'l', col = 'red', main = "ERROR PLOT LINEAR REGRESSION", ylab = "Errors",xlab = "num variale", ylim = c(0.7,1))
lines(error_data4_lin$adj_r_sq,  col = 'green' )
lines(error_data4_lin$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)



## RIDGE


#auto price
data4_ridge1 <- lmridge(price ~ distance + stores, data4, K = c(0.1, 0.001))
data4_ridge2 <- lmridge(price ~ distance + stores + house_age, data4, K = c(0.1, 0.001))
data4_ridge3 <- lmridge(price ~ distance + stores + house_age + latitude, data4, K = c(0.1, 0.001))
data4_ridge4 <- lmridge(price ~ distance + stores + house_age + latitude + tran_date , data4, K = c(0.1, 0.001))
data4_ridge5 <- lmridge(price ~ distance + stores + house_age + latitude + tran_date + longitude, data4, K = c(0.1, 0.001))


cv_data4_ridge1 <- train(price ~ distance + stores, data = data4, trControl = train_control, method = "ridge")
cv_data4_ridge2 <- train(price ~ distance + stores + house_age, data = data4, trControl = train_control, method = "ridge")
cv_data4_ridge3 <- train(price ~ distance + stores + house_age + latitude, data = data4, trControl = train_control, method = "ridge")
cv_data4_ridge4 <- train(price ~ distance + stores + house_age + latitude + tran_date, data = data4, trControl = train_control, method = "ridge")
cv_data4_ridge5 <- train(price ~ distance + stores + house_age + latitude + tran_date + longitude, data = data4, trControl = train_control, method = "ridge")

error_data4_ridge <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data4_ridge <- add_row(error_data4_ridge, r_sq = max(rstats1(data4_ridge1)$R2), adj_r_sq = max(rstats1(data4_ridge1)$adjR2), cv_r_sq = mean(cv_data4_ridge1$resample$Rsquared))
error_data4_ridge <- add_row(error_data4_ridge, r_sq = max(rstats1(data4_ridge2)$R2), adj_r_sq = max(rstats1(data4_ridge2)$adjR2), cv_r_sq = mean(cv_data4_ridge2$resample$Rsquared))
error_data4_ridge <- add_row(error_data4_ridge, r_sq = max(rstats1(data4_ridge3)$R2), adj_r_sq = max(rstats1(data4_ridge3)$adjR2), cv_r_sq = mean(cv_data4_ridge3$resample$Rsquared))  
error_data4_ridge <- add_row(error_data4_ridge, r_sq = max(rstats1(data4_ridge4)$R2), adj_r_sq = max(rstats1(data4_ridge4)$adjR2), cv_r_sq = mean(cv_data4_ridge4$resample$Rsquared))
error_data4_ridge <- add_row(error_data4_ridge, r_sq = max(rstats1(data4_ridge5)$R2), adj_r_sq = max(rstats1(data4_ridge5)$adjR2), cv_r_sq = mean(cv_data4_ridge5$resample$Rsquared))

plot(error_data4_ridge$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RIDGE", ylab = "Errors",xlab = "num variable", ylim = c(0,1))
lines(error_data4_ridge$adj_r_sq,  col = 'green' )
lines(error_data4_ridge$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)



## LASSO

#auto
x <- (data4$distance)
x <- cbind(data4$stores, x)
data4_lasso1 <- lars(x, data4$price, type = 'lasso')

x <- (data4$distance)
x <- cbind(data4$stores, x)
x <- cbind(data4$house_age, x)
data4_lasso2 <- lars(x, data4$price, type = 'lasso')


x <- (data4$distance)
x <- cbind(data4$stores, x)
x <- cbind(data4$house_age, x)
x <- cbind(data4$latitude, x)
data4_lasso3 <- lars(x, data4$price, type = 'lasso')


x <- (data4$distance)
x <- cbind(data4$stores, x)
x <- cbind(data4$house_age, x)
x <- cbind(data4$latitude, x)
x <- cbind(data4$tran_date, x)
data4_lasso4 <- lars(x, data4$price, type = 'lasso')

x <- (data4$distance)
x <- cbind(data4$stores, x)
x <- cbind(data4$house_age, x)
x <- cbind(data4$latitude, x)
x <- cbind(data4$tran_date, x)
x <- cbind(data4$longitude, x)
data4_lasso5 <- lars(x, data4$price, type = 'lasso')



cv_data4_lasso1 <- train(price ~ distance + stores, data = data4, trControl = train_control, method = "lasso")
cv_data4_lasso2 <- train(price ~ distance + stores + house_age, data = data4, trControl = train_control, method = "lasso")
cv_data4_lasso3 <- train(price ~ distance + stores + house_age + latitude, data = data4, trControl = train_control, method = "lasso")
cv_data4_lasso4 <- train(price ~ distance + stores + house_age + latitude + tran_date, data = data4, trControl = train_control, method = "lasso")
cv_data4_lasso5 <- train(price ~ distance + stores + house_age + latitude + tran_date + longitude, data = data4, trControl = train_control, method = "lasso")

error_data4_lasso <- data.frame("r_sq" = double(0), "cv_r_sq" = double(0), )
error_data4_lasso <- add_row(error_data4_lasso, r_sq = max(data4_lasso1$R2), cv_r_sq = mean(cv_data4_lasso1$resample$Rsquared))
error_data4_lasso <- add_row(error_data4_lasso, r_sq = max(data4_lasso2$R2), cv_r_sq = mean(cv_data4_lasso2$resample$Rsquared))
error_data4_lasso <- add_row(error_data4_lasso, r_sq = max(data4_lasso3$R2), cv_r_sq = mean(cv_data4_lasso3$resample$Rsquared))
error_data4_lasso <- add_row(error_data4_lasso, r_sq = max(data4_lasso4$R2), cv_r_sq = mean(cv_data4_lasso4$resample$Rsquared))
error_data4_lasso <- add_row(error_data4_lasso, r_sq = max(data4_lasso5$R2), cv_r_sq = mean(cv_data4_lasso5$resample$Rsquared))

#plot of r square, adjusted r square 
plot(error_data4_lasso$r_sq, type = 'l', col = 'red', main = "ERROR PLOT LASSO", ylab = "Errors",xlab = "num variable", ylim = c(0,1))
lines(error_data4_lasso$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


## QUAD

#transferring the current data into new data frame

##adding square of each of the predictors to the data frame for quad regression
data4$distance_square <- data4$distance^2
data4$stores_square <- data4$stores^2
data4$house_age_square <- data4$house_age^2
data4$latitude_square <- data4$latitude^2
data4$tran_date_square <- data4$tran_date^2
data4$longitude_square <- data4$longitude^2


#regression
data4_quad1 <- lm(price ~ distance + distance_square, data = data4)
data4_quad2 <- lm(price ~ distance + distance_square + stores + stores_square, data = data4)
data4_quad3 <- lm(price ~ distance + distance_square + stores + stores_square + house_age + house_age_square, data = data4)
data4_quad4 <- lm(price ~ distance + distance_square + stores + stores_square + house_age + house_age_square + latitude + latitude_square, data = data4)
data4_quad5 <- lm(price ~ distance + distance_square + stores + stores_square + house_age + house_age_square + latitude + latitude_square + tran_date + tran_date_square, data = data4)
data4_quad6 <- lm(price ~ distance + distance_square + stores + stores_square + house_age + house_age_square + latitude + latitude_square + tran_date + tran_date_square + longitude + longitude_square, data = data4)

#cross validation
cv_data4_quad1 <- train(price ~ distance + distance_square, data = data4, trControl = train_control, method = "lm")
cv_data4_quad2 <- train(price ~ distance + distance_square + stores + stores_square, data = data4, trControl = train_control, method = "lm")
cv_data4_quad3 <- train(price ~ distance + distance_square + stores + stores_square + house_age + house_age_square, data = data4, trControl = train_control, method = "lm")
cv_data4_quad4 <- train(price ~ distance + distance_square + stores + stores_square + house_age + house_age_square + latitude + latitude_square, data = data4, trControl = train_control, method = "lm")
cv_data4_quad5 <- train(price ~ distance + distance_square + stores + stores_square + house_age + house_age_square + latitude + latitude_square + tran_date + tran_date_square, data = data4, trControl = train_control, method = "lm")
cv_data4_quad6 <- train(price ~ distance + distance_square + stores + stores_square + house_age + house_age_square + latitude + latitude_square + tran_date + tran_date_square + longitude + longitude_square, data = data4, trControl = train_control, method = "lm")

#creating a new dataframe to store rsquare adjusted_rsquard and cv rsquared
error_data4_quad <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data4_quad <- add_row(error_data4_quad, r_sq = summary(data4_quad1)$r.squared, adj_r_sq = summary(data4_quad1)$adj.r.squared, cv_r_sq = mean(cv_data4_quad1$resample$Rsquared))
error_data4_quad <- add_row(error_data4_quad, r_sq = summary(data4_quad2)$r.squared, adj_r_sq = summary(data4_quad2)$adj.r.squared, cv_r_sq = mean(cv_data4_quad2$resample$Rsquared))
error_data4_quad <- add_row(error_data4_quad, r_sq = summary(data4_quad3)$r.squared, adj_r_sq = summary(data4_quad3)$adj.r.squared, cv_r_sq = mean(cv_data4_quad3$resample$Rsquared))
error_data4_quad <- add_row(error_data4_quad, r_sq = summary(data4_quad4)$r.squared, adj_r_sq = summary(data4_quad4)$adj.r.squared, cv_r_sq = mean(cv_data4_quad4$resample$Rsquared))
error_data4_quad <- add_row(error_data4_quad, r_sq = summary(data4_quad5)$r.squared, adj_r_sq = summary(data4_quad5)$adj.r.squared, cv_r_sq = mean(cv_data4_quad5$resample$Rsquared))
error_data4_quad <- add_row(error_data4_quad, r_sq = summary(data4_quad6)$r.squared, adj_r_sq = summary(data4_quad6)$adj.r.squared, cv_r_sq = mean(cv_data4_quad6$resample$Rsquared))

#plot of r square, adjusted r square and r square cross-validation
plot(error_data4_quad$r_sq, type = 'l', col = 'red', main = "ERROR PLOT QUAD", ylab = "Errors",xlab = "num variable", ylim = c(0.7,1))
lines(error_data4_quad$adj_r_sq,  col = 'green' )
lines(error_data4_quad$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


## RESPONSE

#creating a new dataframe from auto df for RESPONSE SURFACE REGRESSION
data4 <- data4

data4_rspns1 <- rsm(price ~ SO(distance, stores), data = data4)
data4_rspns2 <- rsm(price ~ SO(distance, stores, house_age), data = data4)
data4_rspns3 <- rsm(price ~ SO(distance, stores, house_age, latitude), data = data4)
data4_rspns4 <- rsm(price ~ SO(distance, stores, house_age, latitude, tran_date), data = data4)
data4_rspns5 <- rsm(price ~ SO(distance, stores, house_age, latitude, tran_date, longitude), data = data4)


error_data4_rspns <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0))
error_data4_rspns <- add_row(error_data4_rspns, r_sq = summary(data4_rspns1)$r.squared, adj_r_sq = summary(data4_rspns1)$adj.r.squared)
error_data4_rspns <- add_row(error_data4_rspns, r_sq = summary(data4_rspns2)$r.squared, adj_r_sq = summary(data4_rspns2)$adj.r.squared)
error_data4_rspns <- add_row(error_data4_rspns, r_sq = summary(data4_rspns3)$r.squared, adj_r_sq = summary(data4_rspns3)$adj.r.squared)
error_data4_rspns <- add_row(error_data4_rspns, r_sq = summary(data4_rspns4)$r.squared, adj_r_sq = summary(data4_rspns4)$adj.r.squared)
error_data4_rspns <- add_row(error_data4_rspns, r_sq = summary(data4_rspns5)$r.squared, adj_r_sq = summary(data4_rspns5)$adj.r.squared)
error_data4_rspns <- add_row(error_data4_rspns, r_sq = summary(data4_rspns6)$r.squared, adj_r_sq = summary(data4_rspns6)$adj.r.squared)

#plot of r square, adjusted r square
plot(error_data4_rspns$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RESPONSE SURFACE", ylab = "Errors", ylim = c(0.3,1))
lines(error_data4_rspns$adj_r_sq,  col = 'green' )
legend(5,0.8, legend = c("R Squared","Adj R Squared"), 
       col = c("red","green"), lty = 1:2, cex = 0.8)



## COMBINED CYCLE POWER PLANT DATA
#reading the data
data5 <- read.xls('filepath')

#replacing missing values with mean
data5$temperature[is.na(data5$temperature)] <- mean(data5$temperature, na.rm = TRUE)
data5$vaccum[is.na(data5$vaccum)] <- mean(data5$vaccum, na.rm = TRUE)
data5$pressure[is.na(data5$pressure)] <- mean(data5$pressure, na.rm = TRUE)
data5$rel_humidity[is.na(data5$rel_humidity)] <- mean(data5$rel_humidity, na.rm = TRUE)
data5$energy_output[is.na(data5$energy_output)] <- mean(data5$energy_output, na.rm = TRUE)

#forward selection for feature selection
fwd_model_auto <- lm(energy_output ~ 1,  data = data5)
step(fwd_model_auto, direction = "forward", scope = formula(energy_output ~ temperature + vaccum + pressure + rel_humidity ))
summary(fwd_model_auto)
#formula obtained energy_output ~ temperature + rel_humidity + vaccum + pressure


##LINEAR REGRESSION
data5_lin1 <- lm(energy_output ~ temperature, data = data5)
data5_lin2 <- lm(energy_output ~ temperature + rel_humidity, data = data5)
data5_lin3 <- lm(energy_output ~ temperature + rel_humidity + vaccum, data = data5)
data5_lin4 <- lm(energy_output ~ temperature + rel_humidity + vaccum + pressure, data = data5)


cv_data5_lin1 <- train(energy_output ~ temperature, data = data5, trControl = train_control, method = "lm")
cv_data5_lin2 <- train(energy_output ~ temperature + rel_humidity, data = data5, trControl = train_control, method = "lm")
cv_data5_lin3 <- train(energy_output ~ temperature + rel_humidity + vaccum, data = data5, trControl = train_control, method = "lm")
cv_data5_lin4 <- train(energy_output ~ temperature + rel_humidity + vaccum + pressure, data = data5, trControl = train_control, method = "lm")


error_data5_lin <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data5_lin <- add_row(error_data5_lin, r_sq = summary(data5_lin1)$r.squared, adj_r_sq = summary(data5_lin1)$adj.r.squared, cv_r_sq = mean(cv_data5_lin1$resample$Rsquared))
error_data5_lin <- add_row(error_data5_lin, r_sq = summary(data5_lin2)$r.squared, adj_r_sq = summary(data5_lin2)$adj.r.squared, cv_r_sq = mean(cv_data5_lin2$resample$Rsquared))
error_data5_lin <- add_row(error_data5_lin, r_sq = summary(data5_lin3)$r.squared, adj_r_sq = summary(data5_lin3)$adj.r.squared, cv_r_sq = mean(cv_data5_lin3$resample$Rsquared))
error_data5_lin <- add_row(error_data5_lin, r_sq = summary(data5_lin4)$r.squared, adj_r_sq = summary(data5_lin4)$adj.r.squared, cv_r_sq = mean(cv_data5_lin4$resample$Rsquared))



#plot of r square, adjusted r square and r square cross-validation
plot(error_data5_lin$r_sq, type = 'l', col = 'red', main = "ERROR PLOT LINEAR REGRESSION", ylab = "Errors",xlab = "num variale", ylim = c(0.7,1))
lines(error_data5_lin$adj_r_sq,  col = 'green' )
lines(error_data5_lin$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


## RIDGE REGRESSION




#auto sound_pressure
data5_lin1 <- lm(sound_pressure ~ freq, data = data5)
data5_lin2 <- lm(sound_pressure ~ freq + displacement, data = data5)
data5_lin3 <- lm(sound_pressure ~ freq + displacement + chord_len, data = data5)
data5_lin4 <- lm(sound_pressure ~ freq + displacement + chord_len + velocity, data = data5)
data5_lin5 <- lm(sound_pressure ~ freq + displacement + chord_len + velocity + angle, data = data5)



cv_data5_lin1 <- train(sound_pressure ~ freq, data = data5, trControl = train_control, method = "lm")
cv_data5_lin2 <- train(sound_pressure ~ freq + displacement, data = data5, trControl = train_control, method = "lm")
cv_data5_lin3 <- train(sound_pressure ~ freq + displacement + chord_len, data = data5, trControl = train_control, method = "lm")
cv_data5_lin4 <- train(sound_pressure ~ freq + displacement + chord_len + velocity, data = data5, trControl = train_control, method = "lm")
cv_data5_lin5 <- train(sound_pressure ~ freq + displacement + chord_len + velocity + angle, data = data5, trControl = train_control, method = "lm")


error_data5_lin <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data5_lin <- add_row(error_data5_lin, r_sq = summary(data5_lin1)$r.squared, adj_r_sq = summary(data5_lin1)$adj.r.squared, cv_r_sq = mean(cv_data5_lin1$resample$Rsquared))
error_data5_lin <- add_row(error_data5_lin, r_sq = summary(data5_lin2)$r.squared, adj_r_sq = summary(data5_lin2)$adj.r.squared, cv_r_sq = mean(cv_data5_lin2$resample$Rsquared))
error_data5_lin <- add_row(error_data5_lin, r_sq = summary(data5_lin3)$r.squared, adj_r_sq = summary(data5_lin3)$adj.r.squared, cv_r_sq = mean(cv_data5_lin3$resample$Rsquared))
error_data5_lin <- add_row(error_data5_lin, r_sq = summary(data5_lin4)$r.squared, adj_r_sq = summary(data5_lin4)$adj.r.squared, cv_r_sq = mean(cv_data5_lin4$resample$Rsquared))
error_data5_lin <- add_row(error_data5_lin, r_sq = summary(data5_lin5)$r.squared, adj_r_sq = summary(data5_lin5)$adj.r.squared, cv_r_sq = mean(cv_data5_lin5$resample$Rsquared))



#plot of r square, adjusted r square and r square cross-validation
plot(error_data5_lin$r_sq, type = 'l', col = 'red', main = "ERROR PLOT LINEAR REGRESSION", ylab = "Errors",xlab = "num variale", ylim = c(0.7,1))
lines(error_data5_lin$adj_r_sq,  col = 'green' )
lines(error_data5_lin$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"),
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)



## RIDGE


#auto sound_pressure
data5_ridge1 <- lmridge(sound_pressure ~ freq + displacement, data5, K = c(0.1, 0.001))
data5_ridge2 <- lmridge(sound_pressure ~ freq + displacement + chord_len, data5, K = c(0.1, 0.001))
data5_ridge3 <- lmridge(sound_pressure ~ freq + displacement + chord_len + velocity, data5, K = c(0.1, 0.001))
data5_ridge4 <- lmridge(sound_pressure ~ freq + displacement + chord_len + velocity + angle, data5, K = c(0.1, 0.001))



cv_data5_ridge1 <- train(sound_pressure ~ freq + displacement, data = data5, trControl = train_control, method = "ridge")
cv_data5_ridge2 <- train(sound_pressure ~ freq + displacement + chord_len, data = data5, trControl = train_control, method = "ridge")
cv_data5_ridge3 <- train(sound_pressure ~ freq + displacement + chord_len + velocity, data = data5, trControl = train_control, method = "ridge")
cv_data5_ridge4 <- train(sound_pressure ~ freq + displacement + chord_len + velocity + angle, data = data5, trControl = train_control, method = "ridge")


error_data5_ridge <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data5_ridge <- add_row(error_data5_ridge, r_sq = max(rstats1(data5_ridge1)$R2), adj_r_sq = max(rstats1(data5_ridge1)$adjR2), cv_r_sq = mean(cv_data5_ridge1$resample$Rsquared))
error_data5_ridge <- add_row(error_data5_ridge, r_sq = max(rstats1(data5_ridge2)$R2), adj_r_sq = max(rstats1(data5_ridge2)$adjR2), cv_r_sq = mean(cv_data5_ridge2$resample$Rsquared))
error_data5_ridge <- add_row(error_data5_ridge, r_sq = max(rstats1(data5_ridge3)$R2), adj_r_sq = max(rstats1(data5_ridge3)$adjR2), cv_r_sq = mean(cv_data5_ridge3$resample$Rsquared))
error_data5_ridge <- add_row(error_data5_ridge, r_sq = max(rstats1(data5_ridge4)$R2), adj_r_sq = max(rstats1(data5_ridge4)$adjR2), cv_r_sq = mean(cv_data5_ridge4$resample$Rsquared))


plot(error_data5_ridge$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RIDGE", ylab = "Errors",xlab = "num variable", ylim = c(0,1))
lines(error_data5_ridge$adj_r_sq,  col = 'green' )
lines(error_data5_ridge$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"),
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)



## LASSO

#auto
x <- (data5$freq)
x <- cbind(data5$displacement, x)
data5_lasso1 <- lars(x, data5$sound_pressure, type = 'lasso')

x <- (data5$freq)
x <- cbind(data5$displacement, x)
x <- cbind(data5$chord_len, x)
data5_lasso2 <- lars(x, data5$sound_pressure, type = 'lasso')


x <- (data5$freq)
x <- cbind(data5$displacement, x)
x <- cbind(data5$chord_len, x)
x <- cbind(data5$velocity, x)
data5_lasso3 <- lars(x, data5$sound_pressure, type = 'lasso')


x <- (data5$freq)
x <- cbind(data5$displacement, x)
x <- cbind(data5$chord_len, x)
x <- cbind(data5$velocity, x)
x <- cbind(data5$angle, x)
data5_lasso4 <- lars(x, data5$sound_pressure, type = 'lasso')



cv_data5_lasso1 <- train(sound_pressure ~ freq + displacement, data = data5, trControl = train_control, method = "lasso")
cv_data5_lasso2 <- train(sound_pressure ~ freq + displacement + chord_len, data = data5, trControl = train_control, method = "lasso")
cv_data5_lasso3 <- train(sound_pressure ~ freq + displacement + chord_len + velocity, data = data5, trControl = train_control, method = "lasso")
cv_data5_lasso4 <- train(sound_pressure ~ freq + displacement + chord_len + velocity + angle, data = data5, trControl = train_control, method = "lasso")

error_data5_lasso <- data.frame("r_sq" = double(0), "cv_r_sq" = double(0))
error_data5_lasso <- add_row(error_data5_lasso, r_sq = max(data5_lasso1$R2), cv_r_sq = mean(cv_data5_lasso1$resample$Rsquared))
error_data5_lasso <- add_row(error_data5_lasso, r_sq = max(data5_lasso2$R2), cv_r_sq = mean(cv_data5_lasso2$resample$Rsquared))
error_data5_lasso <- add_row(error_data5_lasso, r_sq = max(data5_lasso3$R2), cv_r_sq = mean(cv_data5_lasso3$resample$Rsquared))
error_data5_lasso <- add_row(error_data5_lasso, r_sq = max(data5_lasso4$R2), cv_r_sq = mean(cv_data5_lasso4$resample$Rsquared))

#plot of r square, adjusted r square
plot(error_data5_lasso$r_sq, type = 'l', col = 'red', main = "ERROR PLOT LASSO", ylab = "Errors",xlab = "num variable", ylim = c(0,1))
lines(error_data5_lasso$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"),
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


## QUAD

#transferring the current data into new data frame

##adding square of each of the predictors to the data frame for quad regression
data5$freq_square <- data5$freq^2
data5$displacement_square <- data5$displacement^2
data5$chord_len_square <- data5$chord_len^2
data5$velocity_square <- data5$velocity^2
data5$angle_square <- data5$angle^2



#regression
data5_quad1 <- lm(sound_pressure ~ freq + freq_square, data = data5)
data5_quad2 <- lm(sound_pressure ~ freq + freq_square + displacement + displacement_square, data = data5)
data5_quad3 <- lm(sound_pressure ~ freq + freq_square + displacement + displacement_square + chord_len + chord_len_square, data = data5)
data5_quad4 <- lm(sound_pressure ~ freq + freq_square + displacement + displacement_square + chord_len + chord_len_square + velocity + velocity_square, data = data5)
data5_quad5 <- lm(sound_pressure ~ freq + freq_square + displacement + displacement_square + chord_len + chord_len_square + velocity + velocity_square + angle + angle_square, data = data5)

#cross validation
cv_data5_quad1 <- train(sound_pressure ~ freq + freq_square, data = data5, trControl = train_control, method = "lm")
cv_data5_quad2 <- train(sound_pressure ~ freq + freq_square + displacement + displacement_square, data = data5, trControl = train_control, method = "lm")
cv_data5_quad3 <- train(sound_pressure ~ freq + freq_square + displacement + displacement_square + chord_len + chord_len_square, data = data5, trControl = train_control, method = "lm")
cv_data5_quad4 <- train(sound_pressure ~ freq + freq_square + displacement + displacement_square + chord_len + chord_len_square + velocity + velocity_square, data = data5, trControl = train_control, method = "lm")
cv_data5_quad5 <- train(sound_pressure ~ freq + freq_square + displacement + displacement_square + chord_len + chord_len_square + velocity + velocity_square + angle + angle_square, data = data5, trControl = train_control, method = "lm")

#creating a new dataframe to store rsquare adjusted_rsquard and cv rsquared
error_data5_quad <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data5_quad <- add_row(error_data5_quad, r_sq = summary(data5_quad1)$r.squared, adj_r_sq = summary(data5_quad1)$adj.r.squared, cv_r_sq = mean(cv_data5_quad1$resample$Rsquared))
error_data5_quad <- add_row(error_data5_quad, r_sq = summary(data5_quad2)$r.squared, adj_r_sq = summary(data5_quad2)$adj.r.squared, cv_r_sq = mean(cv_data5_quad2$resample$Rsquared))
error_data5_quad <- add_row(error_data5_quad, r_sq = summary(data5_quad3)$r.squared, adj_r_sq = summary(data5_quad3)$adj.r.squared, cv_r_sq = mean(cv_data5_quad3$resample$Rsquared))
error_data5_quad <- add_row(error_data5_quad, r_sq = summary(data5_quad4)$r.squared, adj_r_sq = summary(data5_quad4)$adj.r.squared, cv_r_sq = mean(cv_data5_quad4$resample$Rsquared))
error_data5_quad <- add_row(error_data5_quad, r_sq = summary(data5_quad5)$r.squared, adj_r_sq = summary(data5_quad5)$adj.r.squared, cv_r_sq = mean(cv_data5_quad5$resample$Rsquared))

#plot of r square, adjusted r square and r square cross-validation
plot(error_data5_quad$r_sq, type = 'l', col = 'red', main = "ERROR PLOT QUAD", ylab = "Errors",xlab = "num variable", ylim = c(0.7,1))
lines(error_data5_quad$adj_r_sq,  col = 'green' )
lines(error_data5_quad$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"),
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


## RESPONSE

#creating a new dataframe from auto df for RESPONSE SURFACE REGRESSION
data5 <- data5

data5_rspns1 <- rsm(sound_pressure ~ SO(freq, displacement), data = data5)
data5_rspns2 <- rsm(sound_pressure ~ SO(freq, displacement, chord_len), data = data5)
data5_rspns3 <- rsm(sound_pressure ~ SO(freq, displacement, chord_len, velocity), data = data5)
data5_rspns4 <- rsm(sound_pressure ~ SO(freq, displacement, chord_len, velocity, angle), data = data5)


error_data5_rspns <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0))
error_data5_rspns <- add_row(error_data5_rspns, r_sq = summary(data5_rspns1)$r.squared, adj_r_sq = summary(data5_rspns1)$adj.r.squared)
error_data5_rspns <- add_row(error_data5_rspns, r_sq = summary(data5_rspns2)$r.squared, adj_r_sq = summary(data5_rspns2)$adj.r.squared)
error_data5_rspns <- add_row(error_data5_rspns, r_sq = summary(data5_rspns3)$r.squared, adj_r_sq = summary(data5_rspns3)$adj.r.squared)
error_data5_rspns <- add_row(error_data5_rspns, r_sq = summary(data5_rspns4)$r.squared, adj_r_sq = summary(data5_rspns4)$adj.r.squared)

#plot of r square, adjusted r square
plot(error_data5_rspns$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RESPONSE SURFACE", ylab = "Errors", ylim = c(0.3,1))
lines(error_data5_rspns$adj_r_sq,  col = 'green' )
legend(5,0.8, legend = c("R Squared","Adj R Squared"),
       col = c("red","green"), lty = 1:2, cex = 0.8)



## Computer Hardware Dataset

comp_df <- read.csv('filepath')
data6 <- comp_df

#replacing missing values with mean
data6$myct[is.na(data6$myct)] <- mean(data6$myct, na.rm = TRUE)
data6$mmin[is.na(data6$mmin)] <- mean(data6$mmin, na.rm = TRUE)
data6$mmax[is.na(data6$mmax)] <- mean(data6$mmax, na.rm = TRUE)
data6$cach[is.na(data6$cach)] <- mean(data6$cach, na.rm = TRUE)
data6$chmin[is.na(data6$chmin)] <- mean(data6$chmin, na.rm = TRUE)
data6$chmax[is.na(data6$chmax)] <- mean(data6$chmax, na.rm = TRUE)
data6$prp[is.na(data6$prp)] <- mean(data6$prp, na.rm = TRUE)
data6$erp[is.na(data6$erp)] <- mean(data6$erp, na.rm = TRUE)


# forward selection
fwd_model_auto <- lm(erp ~ 1,  data = data6)
step(fwd_model_auto, direction = "forward", scope = formula(erp ~ myct + mmin + mmax + cach + chmin + chmax + prp))
summary(fwd_model_auto)

train_control <- trainControl(method = "cv", number = 10)


#auto erp
data6_lin1 <- lm(erp ~ prp, data = data6)
data6_lin2 <- lm(erp ~ prp + mmax, data = data6)
data6_lin3 <- lm(erp ~ prp + mmax + mmin, data = data6)
data6_lin4 <- lm(erp ~ prp + mmax + mmin + myct, data = data6)
data6_lin5 <- lm(erp ~ prp + mmax + mmin + myct + chmax, data = data6)
data6_lin6 <- lm(erp ~ prp + mmax + mmin + myct + chmax + cach, data = data6)
data6_lin7 <- lm(erp ~ prp + mmax + mmin + myct + chmax + cach + chmin, data = data6)


#cross validation
cv_data6_lin1 <- train(erp ~ prp, data = data6, trControl = train_control, method = "lm")
cv_data6_lin2 <- train(erp ~ prp + mmax, data = data6, trControl = train_control, method = "lm")
cv_data6_lin3 <- train(erp ~ prp + mmax + mmin, data = data6, trControl = train_control, method = "lm")
cv_data6_lin4 <- train(erp ~ prp + mmax + mmin + myct, data = data6, trControl = train_control, method = "lm")
cv_data6_lin5 <- train(erp ~ prp + mmax + mmin + myct + chmax, data = data6, trControl = train_control, method = "lm")
cv_data6_lin6 <- train(erp ~ prp + mmax + mmin + myct + chmax + cach, data = data6, trControl = train_control, method = "lm")
cv_data6_lin7 <- train(erp ~ prp + mmax + mmin + myct + chmax + cach + chmin, data = data6, trControl = train_control, method = "lm")


#storing r square, adj r square and cv r square in dataframe
error_data6_lin <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data6_lin <- add_row(error_data6_lin, r_sq = summary(data6_lin1)$r.squared, adj_r_sq = summary(data6_lin1)$adj.r.squared, cv_r_sq = mean(cv_data6_lin1$resample$Rsquared))
error_data6_lin <- add_row(error_data6_lin, r_sq = summary(data6_lin2)$r.squared, adj_r_sq = summary(data6_lin2)$adj.r.squared, cv_r_sq = mean(cv_data6_lin2$resample$Rsquared))
error_data6_lin <- add_row(error_data6_lin, r_sq = summary(data6_lin3)$r.squared, adj_r_sq = summary(data6_lin3)$adj.r.squared, cv_r_sq = mean(cv_data6_lin3$resample$Rsquared))
error_data6_lin <- add_row(error_data6_lin, r_sq = summary(data6_lin4)$r.squared, adj_r_sq = summary(data6_lin4)$adj.r.squared, cv_r_sq = mean(cv_data6_lin4$resample$Rsquared))
error_data6_lin <- add_row(error_data6_lin, r_sq = summary(data6_lin5)$r.squared, adj_r_sq = summary(data6_lin5)$adj.r.squared, cv_r_sq = mean(cv_data6_lin5$resample$Rsquared))
error_data6_lin <- add_row(error_data6_lin, r_sq = summary(data6_lin6)$r.squared, adj_r_sq = summary(data6_lin6)$adj.r.squared, cv_r_sq = mean(cv_data6_lin6$resample$Rsquared))
error_data6_lin <- add_row(error_data6_lin, r_sq = summary(data6_lin7)$r.squared, adj_r_sq = summary(data6_lin7)$adj.r.squared, cv_r_sq = mean(cv_data6_lin7$resample$Rsquared))



#plot of r square, adjusted r square and r square cross-validation
plot(error_data6_lin$r_sq, type = 'l', col = 'red', main = "ERROR PLOT LINEAR REGRESSION", ylab = "Errors",xlab = "num variale", ylim = c(0,1))
lines(error_data6_lin$adj_r_sq,  col = 'green' )
lines(error_data6_lin$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"),
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)



## RIDGE


#ridge regression
data6_ridge1 <- lmridge(erp ~ prp + mmax, data6, K = c(0.1, 0.001))
data6_ridge2 <- lmridge(erp ~ prp + mmax + mmin, data6, K = c(0.1, 0.001))
data6_ridge3 <- lmridge(erp ~ prp + mmax + mmin + myct, data6, K = c(0.1, 0.001))
data6_ridge4 <- lmridge(erp ~ prp + mmax + mmin + myct + chmax , data6, K = c(0.1, 0.001))
data6_ridge5 <- lmridge(erp ~ prp + mmax + mmin + myct + chmax + cach, data6, K = c(0.1, 0.001))
data6_ridge6 <- lmridge(erp ~ prp + mmax + mmin + myct + chmax + cach + chmin, data6, K = c(0.1, 0.001))

#cross validation
cv_data6_ridge1 <- train(erp ~ prp + mmax, data = data6, trControl = train_control, method = "ridge")
cv_data6_ridge2 <- train(erp ~ prp + mmax + mmin, data = data6, trControl = train_control, method = "ridge")
cv_data6_ridge3 <- train(erp ~ prp + mmax + mmin + myct, data = data6, trControl = train_control, method = "ridge")
cv_data6_ridge4 <- train(erp ~ prp + mmax + mmin + myct + chmax, data = data6, trControl = train_control, method = "ridge")
cv_data6_ridge5 <- train(erp ~ prp + mmax + mmin + myct + chmax + cach, data = data6, trControl = train_control, method = "ridge")
cv_data6_ridge6 <- train(erp ~ prp + mmax + mmin + myct + chmax + cach + chmin, data = data6, trControl = train_control, method = "ridge")


error_data6_ridge <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data6_ridge <- add_row(error_data6_ridge, r_sq = max(rstats1(data6_ridge1)$R2), adj_r_sq = max(rstats1(data6_ridge1)$adjR2), cv_r_sq = mean(cv_data6_ridge1$resample$Rsquared))
error_data6_ridge <- add_row(error_data6_ridge, r_sq = max(rstats1(data6_ridge2)$R2), adj_r_sq = max(rstats1(data6_ridge2)$adjR2), cv_r_sq = mean(cv_data6_ridge2$resample$Rsquared))
error_data6_ridge <- add_row(error_data6_ridge, r_sq = max(rstats1(data6_ridge3)$R2), adj_r_sq = max(rstats1(data6_ridge3)$adjR2), cv_r_sq = mean(cv_data6_ridge3$resample$Rsquared))
error_data6_ridge <- add_row(error_data6_ridge, r_sq = max(rstats1(data6_ridge4)$R2), adj_r_sq = max(rstats1(data6_ridge4)$adjR2), cv_r_sq = mean(cv_data6_ridge4$resample$Rsquared))
error_data6_ridge <- add_row(error_data6_ridge, r_sq = max(rstats1(data6_ridge5)$R2), adj_r_sq = max(rstats1(data6_ridge5)$adjR2), cv_r_sq = mean(cv_data6_ridge5$resample$Rsquared))
error_data6_ridge <- add_row(error_data6_ridge, r_sq = max(rstats1(data6_ridge6)$R2), adj_r_sq = max(rstats1(data6_ridge6)$adjR2), cv_r_sq = mean(cv_data6_ridge6$resample$Rsquared))


plot(error_data6_ridge$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RIDGE", ylab = "Errors",xlab = "num variable", ylim = c(0,1))
lines(error_data6_ridge$adj_r_sq,  col = 'green' )
lines(error_data6_ridge$cv_r_sq,  col = 'blue')
legend(1,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"),
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)



## LASSO

#lasso
x <- (data6$prp)
x <- cbind(data6$mmax, x)
data6_lasso1 <- lars(x, data6$erp, type = 'lasso')

x <- (data6$prp)
x <- cbind(data6$mmax, x)
x <- cbind(data6$mmin, x)
data6_lasso2 <- lars(x, data6$erp, type = 'lasso')


x <- (data6$prp)
x <- cbind(data6$mmax, x)
x <- cbind(data6$mmin, x)
x <- cbind(data6$myct, x)
data6_lasso3 <- lars(x, data6$erp, type = 'lasso')


x <- (data6$prp)
x <- cbind(data6$mmax, x)
x <- cbind(data6$mmin, x)
x <- cbind(data6$myct, x)
x <- cbind(data6$chmax, x)
data6_lasso4 <- lars(x, data6$erp, type = 'lasso')

x <- (data6$prp)
x <- cbind(data6$mmax, x)
x <- cbind(data6$mmin, x)
x <- cbind(data6$myct, x)
x <- cbind(data6$chmax, x)
x <- cbind(data6$cach, x)
data6_lasso5 <- lars(x, data6$erp, type = 'lasso')

x <- (data6$prp)
x <- cbind(data6$mmax, x)
x <- cbind(data6$mmin, x)
x <- cbind(data6$myct, x)
x <- cbind(data6$chmax, x)
x <- cbind(data6$cach, x)
x <- cbind(data6$chmin, x)
data6_lasso6 <- lars(x, data6$erp, type = 'lasso')



cv_data6_lasso1 <- train(erp ~ prp + mmax, data = data6, trControl = train_control, method = "lasso")
cv_data6_lasso2 <- train(erp ~ prp + mmax + mmin, data = data6, trControl = train_control, method = "lasso")
cv_data6_lasso3 <- train(erp ~ prp + mmax + mmin + myct, data = data6, trControl = train_control, method = "lasso")
cv_data6_lasso4 <- train(erp ~ prp + mmax + mmin + myct + chmax, data = data6, trControl = train_control, method = "lasso")
cv_data6_lasso5 <- train(erp ~ prp + mmax + mmin + myct + chmax + cach, data = data6, trControl = train_control, method = "lasso")
cv_data6_lasso6 <- train(erp ~ prp + mmax + mmin + myct + chmax + cach + chmin, data = data6, trControl = train_control, method = "lasso")


error_data6_lasso <- data.frame("r_sq" = double(0), "cv_r_sq" = double(0) )
error_data6_lasso <- add_row(error_data6_lasso, r_sq = max(data6_lasso1$R2), cv_r_sq = mean(cv_data6_lasso1$resample$Rsquared))
error_data6_lasso <- add_row(error_data6_lasso, r_sq = max(data6_lasso2$R2), cv_r_sq = mean(cv_data6_lasso2$resample$Rsquared))
error_data6_lasso <- add_row(error_data6_lasso, r_sq = max(data6_lasso3$R2), cv_r_sq = mean(cv_data6_lasso3$resample$Rsquared))
error_data6_lasso <- add_row(error_data6_lasso, r_sq = max(data6_lasso4$R2), cv_r_sq = mean(cv_data6_lasso4$resample$Rsquared))
error_data6_lasso <- add_row(error_data6_lasso, r_sq = max(data6_lasso5$R2), cv_r_sq = mean(cv_data6_lasso5$resample$Rsquared))
error_data6_lasso <- add_row(error_data6_lasso, r_sq = max(data6_lasso6$R2), cv_r_sq = mean(cv_data6_lasso6$resample$Rsquared))


#plot of r square, adjusted r square
plot(error_data6_lasso$r_sq, type = 'l', col = 'red', main = "ERROR PLOT LASSO", ylab = "Errors",xlab = "num variable", ylim = c(0,1))
lines(error_data6_lasso$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"),
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


## QUAD

#transferring the current data into new data frame

##adding square of each of the predictors to the data frame for quad regression
data6$prp_square <- data6$prp^2
data6$mmax_square <- data6$mmax^2
data6$mmin_square <- data6$mmin^2
data6$myct_square <- data6$myct^2
data6$chmax_square <- data6$chmax^2
data6$cach_square <- data6$cach^2
data6$chmin_square <- data6$chmin^2


#regression
data6_quad1 <- lm(erp ~ prp + prp_square, data = data6)
data6_quad2 <- lm(erp ~ prp + prp_square + mmax + mmax_square, data = data6)
data6_quad3 <- lm(erp ~ prp + prp_square + mmax + mmax_square + mmin + mmin_square, data = data6)
data6_quad4 <- lm(erp ~ prp + prp_square + mmax + mmax_square + mmin + mmin_square + myct + myct_square, data = data6)
data6_quad5 <- lm(erp ~ prp + prp_square + mmax + mmax_square + mmin + mmin_square + myct + myct_square + chmax + chmax_square, data = data6)
data6_quad6 <- lm(erp ~ prp + prp_square + mmax + mmax_square + mmin + mmin_square + myct + myct_square + chmax + chmax_square + cach + cach_square, data = data6)
data6_quad7 <- lm(erp ~ prp + prp_square + mmax + mmax_square + mmin + mmin_square + myct + myct_square + chmax + chmax_square + cach + cach_square + chmin + chmin_square, data = data6)


#cross validation
cv_data6_quad1 <- train(erp ~ prp + prp_square, data = data6, trControl = train_control, method = "lm")
cv_data6_quad2 <- train(erp ~ prp + prp_square + mmax + mmax_square, data = data6, trControl = train_control, method = "lm")
cv_data6_quad3 <- train(erp ~ prp + prp_square + mmax + mmax_square + mmin + mmin_square, data = data6, trControl = train_control, method = "lm")
cv_data6_quad4 <- train(erp ~ prp + prp_square + mmax + mmax_square + mmin + mmin_square + myct + myct_square, data = data6, trControl = train_control, method = "lm")
cv_data6_quad5 <- train(erp ~ prp + prp_square + mmax + mmax_square + mmin + mmin_square + myct + myct_square + chmax + chmax_square, data = data6, trControl = train_control, method = "lm")
cv_data6_quad6 <- train(erp ~ prp + prp_square + mmax + mmax_square + mmin + mmin_square + myct + myct_square + chmax + chmax_square + cach + cach_square, data = data6, trControl = train_control, method = "lm")
cv_data6_quad7 <- train(erp ~ prp + prp_square + mmax + mmax_square + mmin + mmin_square + myct + myct_square + chmax + chmax_square + cach + cach_square + chmin + chmin_square, data = data6, trControl = train_control, method = "lm")


#creating a new dataframe to store rsquare adjusted_rsquard and cv rsquared
error_data6_quad <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data6_quad <- add_row(error_data6_quad, r_sq = summary(data6_quad1)$r.squared, adj_r_sq = summary(data6_quad1)$adj.r.squared, cv_r_sq = mean(cv_data6_quad1$resample$Rsquared))
error_data6_quad <- add_row(error_data6_quad, r_sq = summary(data6_quad2)$r.squared, adj_r_sq = summary(data6_quad2)$adj.r.squared, cv_r_sq = mean(cv_data6_quad2$resample$Rsquared))
error_data6_quad <- add_row(error_data6_quad, r_sq = summary(data6_quad3)$r.squared, adj_r_sq = summary(data6_quad3)$adj.r.squared, cv_r_sq = mean(cv_data6_quad3$resample$Rsquared))
error_data6_quad <- add_row(error_data6_quad, r_sq = summary(data6_quad4)$r.squared, adj_r_sq = summary(data6_quad4)$adj.r.squared, cv_r_sq = mean(cv_data6_quad4$resample$Rsquared))
error_data6_quad <- add_row(error_data6_quad, r_sq = summary(data6_quad5)$r.squared, adj_r_sq = summary(data6_quad5)$adj.r.squared, cv_r_sq = mean(cv_data6_quad5$resample$Rsquared))
error_data6_quad <- add_row(error_data6_quad, r_sq = summary(data6_quad6)$r.squared, adj_r_sq = summary(data6_quad6)$adj.r.squared, cv_r_sq = mean(cv_data6_quad6$resample$Rsquared))
error_data6_quad <- add_row(error_data6_quad, r_sq = summary(data6_quad7)$r.squared, adj_r_sq = summary(data6_quad7)$adj.r.squared, cv_r_sq = mean(cv_data6_quad7$resample$Rsquared))


#plot of r square, adjusted r square and r square cross-validation
plot(error_data6_quad$r_sq, type = 'l', col = 'red', main = "ERROR PLOT QUAD", ylab = "Errors",xlab = "num variable", ylim = c(0,1))
lines(error_data6_quad$adj_r_sq,  col = 'green' )
lines(error_data6_quad$cv_r_sq,  col = 'blue')
legend(1,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"),
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


## RESPONSE

#creating a new dataframe from auto df for RESPONSE SURFACE REGRESSION
data6 <- data6

data6_rspns1 <- rsm(erp ~ SO(prp, mmax), data = data6)
data6_rspns2 <- rsm(erp ~ SO(prp, mmax, mmin), data = data6)
data6_rspns3 <- rsm(erp ~ SO(prp, mmax, mmin, myct), data = data6)
data6_rspns4 <- rsm(erp ~ SO(prp, mmax, mmin, myct, chmax), data = data6)
data6_rspns5 <- rsm(erp ~ SO(prp, mmax, mmin, myct, chmax, cach), data = data6)
data6_rspns6 <- rsm(erp ~ SO(prp, mmax, mmin, myct, chmax, cach, chmin), data = data6)



error_data6_rspns <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0))
error_data6_rspns <- add_row(error_data6_rspns, r_sq = summary(data6_rspns1)$r.squared, adj_r_sq = summary(data6_rspns1)$adj.r.squared)
error_data6_rspns <- add_row(error_data6_rspns, r_sq = summary(data6_rspns2)$r.squared, adj_r_sq = summary(data6_rspns2)$adj.r.squared)
error_data6_rspns <- add_row(error_data6_rspns, r_sq = summary(data6_rspns3)$r.squared, adj_r_sq = summary(data6_rspns3)$adj.r.squared)
error_data6_rspns <- add_row(error_data6_rspns, r_sq = summary(data6_rspns4)$r.squared, adj_r_sq = summary(data6_rspns4)$adj.r.squared)
error_data6_rspns <- add_row(error_data6_rspns, r_sq = summary(data6_rspns5)$r.squared, adj_r_sq = summary(data6_rspns5)$adj.r.squared)
error_data6_rspns <- add_row(error_data6_rspns, r_sq = summary(data6_rspns6)$r.squared, adj_r_sq = summary(data6_rspns6)$adj.r.squared)


#plot of r square, adjusted r square
plot(error_data6_rspns$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RESPONSE SURFACE", ylab = "Errors", ylim = c(0.3,1))
lines(error_data6_rspns$adj_r_sq,  col = 'green' )
legend(5,0.8, legend = c("R Squared","Adj R Squared"),
       col = c("red","green"), lty = 1:2, cex = 0.8)


## GPS DATASET

data8$speed[is.na(data8$speed)] <- mean(data8$speed, na.rm = TRUE)
data8$time[is.na(data8$time)] <- mean(data8$time, na.rm = TRUE)
data8$distance[is.na(data8$distance)] <- mean(data8$distance, na.rm = TRUE)
data8$rating[is.na(data8$rating)] <- mean(data8$rating, na.rm = TRUE)
data8$rating_bus[is.na(data8$rating_bus)] <- mean(data8$rating_bus, na.rm = TRUE)
data8$rating_weather[is.na(data8$rating_weather)] <- mean(data8$rating_weather, na.rm = TRUE)
data8$car_or_bus[is.na(data8$car_or_bus)] <- mean(data8$car_or_bus, na.rm = TRUE)
data8$id_android[is.na(data8$id_android)] <- mean(data8$id_android, na.rm = TRUE)

fwd_model_auto <- lm(id_android ~ 1,  data = data8)
step(fwd_model_auto, direction = "forward", scope = formula(id_android ~ speed + time + distance + rating + rating_bus + rating_weather + car_or_bus))
summary(fwd_model_auto)

#formula obtained id_android ~ rating_weather + car_or_bus +   rest(speed + time + rating_bus + distance + rating)

train_control <- trainControl(method = "cv", number = 10)


#auto id_android
data8_lin1 <- lm(id_android ~ rating_weather, data = data8)
data8_lin2 <- lm(id_android ~ rating_weather + car_or_bus, data = data8)
data8_lin3 <- lm(id_android ~ rating_weather + car_or_bus + speed, data = data8)
data8_lin4 <- lm(id_android ~ rating_weather + car_or_bus + speed + time, data = data8)
data8_lin5 <- lm(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus, data = data8)
data8_lin6 <- lm(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus + distance, data = data8)
data8_lin7 <- lm(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus + distance + rating, data = data8)


#cross validation
cv_data8_lin1 <- train(id_android ~ rating_weather, data = data8, trControl = train_control, method = "lm")
cv_data8_lin2 <- train(id_android ~ rating_weather + car_or_bus, data = data8, trControl = train_control, method = "lm")
cv_data8_lin3 <- train(id_android ~ rating_weather + car_or_bus + speed, data = data8, trControl = train_control, method = "lm")
cv_data8_lin4 <- train(id_android ~ rating_weather + car_or_bus + speed + time, data = data8, trControl = train_control, method = "lm")
cv_data8_lin5 <- train(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus, data = data8, trControl = train_control, method = "lm")
cv_data8_lin6 <- train(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus + distance, data = data8, trControl = train_control, method = "lm")
cv_data8_lin7 <- train(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus + distance + rating, data = data8, trControl = train_control, method = "lm")


#storing r square, adj r square and cv r square in dataframe
error_data8_lin <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data8_lin <- add_row(error_data8_lin, r_sq = summary(data8_lin1)$r.squared, adj_r_sq = summary(data8_lin1)$adj.r.squared, cv_r_sq = mean(cv_data8_lin1$resample$Rsquared))
error_data8_lin <- add_row(error_data8_lin, r_sq = summary(data8_lin2)$r.squared, adj_r_sq = summary(data8_lin2)$adj.r.squared, cv_r_sq = mean(cv_data8_lin2$resample$Rsquared))
error_data8_lin <- add_row(error_data8_lin, r_sq = summary(data8_lin3)$r.squared, adj_r_sq = summary(data8_lin3)$adj.r.squared, cv_r_sq = mean(cv_data8_lin3$resample$Rsquared))
error_data8_lin <- add_row(error_data8_lin, r_sq = summary(data8_lin4)$r.squared, adj_r_sq = summary(data8_lin4)$adj.r.squared, cv_r_sq = mean(cv_data8_lin4$resample$Rsquared))
error_data8_lin <- add_row(error_data8_lin, r_sq = summary(data8_lin5)$r.squared, adj_r_sq = summary(data8_lin5)$adj.r.squared, cv_r_sq = mean(cv_data8_lin5$resample$Rsquared))
error_data8_lin <- add_row(error_data8_lin, r_sq = summary(data8_lin6)$r.squared, adj_r_sq = summary(data8_lin6)$adj.r.squared, cv_r_sq = mean(cv_data8_lin6$resample$Rsquared))
error_data8_lin <- add_row(error_data8_lin, r_sq = summary(data8_lin7)$r.squared, adj_r_sq = summary(data8_lin7)$adj.r.squared, cv_r_sq = mean(cv_data8_lin7$resample$Rsquared))



#plot of r square, adjusted r square and r square cross-validation
plot(error_data8_lin$r_sq, type = 'l', col = 'red', main = "ERROR PLOT LINEAR REGRESSION", ylab = "Errors",xlab = "num variale", ylim = c(0,1))
lines(error_data8_lin$adj_r_sq,  col = 'green' )
lines(error_data8_lin$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"),
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)



## RIDGE


#ridge regression
data8_ridge1 <- lmridge(id_android ~ rating_weather + car_or_bus, data8, K = c(0.1, 0.001))
data8_ridge2 <- lmridge(id_android ~ rating_weather + car_or_bus + speed, data8, K = c(0.1, 0.001))
data8_ridge3 <- lmridge(id_android ~ rating_weather + car_or_bus + speed + time, data8, K = c(0.1, 0.001))
data8_ridge4 <- lmridge(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus , data8, K = c(0.1, 0.001))
data8_ridge5 <- lmridge(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus + distance, data8, K = c(0.1, 0.001))
data8_ridge6 <- lmridge(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus + distance + rating, data8, K = c(0.1, 0.001))

#cross validation
cv_data8_ridge1 <- train(id_android ~ rating_weather + car_or_bus, data = data8, trControl = train_control, method = "ridge")
cv_data8_ridge2 <- train(id_android ~ rating_weather + car_or_bus + speed, data = data8, trControl = train_control, method = "ridge")
cv_data8_ridge3 <- train(id_android ~ rating_weather + car_or_bus + speed + time, data = data8, trControl = train_control, method = "ridge")
cv_data8_ridge4 <- train(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus, data = data8, trControl = train_control, method = "ridge")
cv_data8_ridge5 <- train(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus + distance, data = data8, trControl = train_control, method = "ridge")
cv_data8_ridge6 <- train(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus + distance + rating, data = data8, trControl = train_control, method = "ridge")


error_data8_ridge <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data8_ridge <- add_row(error_data8_ridge, r_sq = max(rstats1(data8_ridge1)$R2), adj_r_sq = max(rstats1(data8_ridge1)$adjR2), cv_r_sq = mean(cv_data8_ridge1$resample$Rsquared))
error_data8_ridge <- add_row(error_data8_ridge, r_sq = max(rstats1(data8_ridge2)$R2), adj_r_sq = max(rstats1(data8_ridge2)$adjR2), cv_r_sq = mean(cv_data8_ridge2$resample$Rsquared))
error_data8_ridge <- add_row(error_data8_ridge, r_sq = max(rstats1(data8_ridge3)$R2), adj_r_sq = max(rstats1(data8_ridge3)$adjR2), cv_r_sq = mean(cv_data8_ridge3$resample$Rsquared))
error_data8_ridge <- add_row(error_data8_ridge, r_sq = max(rstats1(data8_ridge4)$R2), adj_r_sq = max(rstats1(data8_ridge4)$adjR2), cv_r_sq = mean(cv_data8_ridge4$resample$Rsquared))
error_data8_ridge <- add_row(error_data8_ridge, r_sq = max(rstats1(data8_ridge5)$R2), adj_r_sq = max(rstats1(data8_ridge5)$adjR2), cv_r_sq = mean(cv_data8_ridge5$resample$Rsquared))
error_data8_ridge <- add_row(error_data8_ridge, r_sq = max(rstats1(data8_ridge6)$R2), adj_r_sq = max(rstats1(data8_ridge6)$adjR2), cv_r_sq = mean(cv_data8_ridge6$resample$Rsquared))


plot(error_data8_ridge$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RIDGE", ylab = "Errors",xlab = "num variable", ylim = c(0,1))
lines(error_data8_ridge$adj_r_sq,  col = 'green' )
lines(error_data8_ridge$cv_r_sq,  col = 'blue')
legend(1,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"),
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)



## LASSO

#lasso
x <- (data8$rating_weather)
x <- cbind(data8$car_or_bus, x)
data8_lasso1 <- lars(x, data8$id_android, type = 'lasso')

x <- (data8$rating_weather)
x <- cbind(data8$car_or_bus, x)
x <- cbind(data8$speed, x)
data8_lasso2 <- lars(x, data8$id_android, type = 'lasso')


x <- (data8$rating_weather)
x <- cbind(data8$car_or_bus, x)
x <- cbind(data8$speed, x)
x <- cbind(data8$time, x)
data8_lasso3 <- lars(x, data8$id_android, type = 'lasso')


x <- (data8$rating_weather)
x <- cbind(data8$car_or_bus, x)
x <- cbind(data8$speed, x)
x <- cbind(data8$time, x)
x <- cbind(data8$rating_bus, x)
data8_lasso4 <- lars(x, data8$id_android, type = 'lasso')

x <- (data8$rating_weather)
x <- cbind(data8$car_or_bus, x)
x <- cbind(data8$speed, x)
x <- cbind(data8$time, x)
x <- cbind(data8$rating_bus, x)
x <- cbind(data8$distance, x)
data8_lasso5 <- lars(x, data8$id_android, type = 'lasso')

x <- (data8$rating_weather)
x <- cbind(data8$car_or_bus, x)
x <- cbind(data8$speed, x)
x <- cbind(data8$time, x)
x <- cbind(data8$rating_bus, x)
x <- cbind(data8$distance, x)
x <- cbind(data8$rating, x)
data8_lasso6 <- lars(x, data8$id_android, type = 'lasso')



cv_data8_lasso1 <- train(id_android ~ rating_weather + car_or_bus, data = data8, trControl = train_control, method = "lasso")
cv_data8_lasso2 <- train(id_android ~ rating_weather + car_or_bus + speed, data = data8, trControl = train_control, method = "lasso")
cv_data8_lasso3 <- train(id_android ~ rating_weather + car_or_bus + speed + time, data = data8, trControl = train_control, method = "lasso")
cv_data8_lasso4 <- train(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus, data = data8, trControl = train_control, method = "lasso")
cv_data8_lasso5 <- train(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus + distance, data = data8, trControl = train_control, method = "lasso")
cv_data8_lasso6 <- train(id_android ~ rating_weather + car_or_bus + speed + time + rating_bus + distance + rating, data = data8, trControl = train_control, method = "lasso")


error_data8_lasso <- data.frame("r_sq" = double(0), "cv_r_sq" = double(0) )
error_data8_lasso <- add_row(error_data8_lasso, r_sq = max(data8_lasso1$R2), cv_r_sq = mean(cv_data8_lasso1$resample$Rsquared))
error_data8_lasso <- add_row(error_data8_lasso, r_sq = max(data8_lasso2$R2), cv_r_sq = mean(cv_data8_lasso2$resample$Rsquared))
error_data8_lasso <- add_row(error_data8_lasso, r_sq = max(data8_lasso3$R2), cv_r_sq = mean(cv_data8_lasso3$resample$Rsquared))
error_data8_lasso <- add_row(error_data8_lasso, r_sq = max(data8_lasso4$R2), cv_r_sq = mean(cv_data8_lasso4$resample$Rsquared))
error_data8_lasso <- add_row(error_data8_lasso, r_sq = max(data8_lasso5$R2), cv_r_sq = mean(cv_data8_lasso5$resample$Rsquared))
error_data8_lasso <- add_row(error_data8_lasso, r_sq = max(data8_lasso6$R2), cv_r_sq = mean(cv_data8_lasso6$resample$Rsquared))


#plot of r square, adjusted r square
plot(error_data8_lasso$r_sq, type = 'l', col = 'red', main = "ERROR PLOT LASSO", ylab = "Errors",xlab = "num variable", ylim = c(0,1))
lines(error_data8_lasso$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"),
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


## QUAD

#transferring the current data into new data frame

##adding square of each of the predictors to the data frame for quad regression
data8$rating_weather_square <- data8$rating_weather^2
data8$car_or_bus_square <- data8$car_or_bus^2
data8$speed_square <- data8$speed^2
data8$time_square <- data8$time^2
data8$rating_bus_square <- data8$rating_bus^2
data8$distance_square <- data8$distance^2
data8$rating_square <- data8$rating^2


#regression
data8_quad1 <- lm(id_android ~ rating_weather + rating_weather_square, data = data8)
data8_quad2 <- lm(id_android ~ rating_weather + rating_weather_square + car_or_bus + car_or_bus_square, data = data8)
data8_quad3 <- lm(id_android ~ rating_weather + rating_weather_square + car_or_bus + car_or_bus_square + speed + speed_square, data = data8)
data8_quad4 <- lm(id_android ~ rating_weather + rating_weather_square + car_or_bus + car_or_bus_square + speed + speed_square + time + time_square, data = data8)
data8_quad5 <- lm(id_android ~ rating_weather + rating_weather_square + car_or_bus + car_or_bus_square + speed + speed_square + time + time_square + rating_bus + rating_bus_square, data = data8)
data8_quad6 <- lm(id_android ~ rating_weather + rating_weather_square + car_or_bus + car_or_bus_square + speed + speed_square + time + time_square + rating_bus + rating_bus_square + distance + distance_square, data = data8)
data8_quad7 <- lm(id_android ~ rating_weather + rating_weather_square + car_or_bus + car_or_bus_square + speed + speed_square + time + time_square + rating_bus + rating_bus_square + distance + distance_square + rating + rating_square, data = data8)


#cross validation
cv_data8_quad1 <- train(id_android ~ rating_weather + rating_weather_square, data = data8, trControl = train_control, method = "lm")
cv_data8_quad2 <- train(id_android ~ rating_weather + rating_weather_square + car_or_bus + car_or_bus_square, data = data8, trControl = train_control, method = "lm")
cv_data8_quad3 <- train(id_android ~ rating_weather + rating_weather_square + car_or_bus + car_or_bus_square + speed + speed_square, data = data8, trControl = train_control, method = "lm")
cv_data8_quad4 <- train(id_android ~ rating_weather + rating_weather_square + car_or_bus + car_or_bus_square + speed + speed_square + time + time_square, data = data8, trControl = train_control, method = "lm")
cv_data8_quad5 <- train(id_android ~ rating_weather + rating_weather_square + car_or_bus + car_or_bus_square + speed + speed_square + time + time_square + rating_bus + rating_bus_square, data = data8, trControl = train_control, method = "lm")
cv_data8_quad6 <- train(id_android ~ rating_weather + rating_weather_square + car_or_bus + car_or_bus_square + speed + speed_square + time + time_square + rating_bus + rating_bus_square + distance + distance_square, data = data8, trControl = train_control, method = "lm")
cv_data8_quad7 <- train(id_android ~ rating_weather + rating_weather_square + car_or_bus + car_or_bus_square + speed + speed_square + time + time_square + rating_bus + rating_bus_square + distance + distance_square + rating + rating_square, data = data8, trControl = train_control, method = "lm")


#creating a new dataframe to store rsquare adjusted_rsquard and cv rsquared
error_data8_quad <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_data8_quad <- add_row(error_data8_quad, r_sq = summary(data8_quad1)$r.squared, adj_r_sq = summary(data8_quad1)$adj.r.squared, cv_r_sq = mean(cv_data8_quad1$resample$Rsquared))
error_data8_quad <- add_row(error_data8_quad, r_sq = summary(data8_quad2)$r.squared, adj_r_sq = summary(data8_quad2)$adj.r.squared, cv_r_sq = mean(cv_data8_quad2$resample$Rsquared))
error_data8_quad <- add_row(error_data8_quad, r_sq = summary(data8_quad3)$r.squared, adj_r_sq = summary(data8_quad3)$adj.r.squared, cv_r_sq = mean(cv_data8_quad3$resample$Rsquared))
error_data8_quad <- add_row(error_data8_quad, r_sq = summary(data8_quad4)$r.squared, adj_r_sq = summary(data8_quad4)$adj.r.squared, cv_r_sq = mean(cv_data8_quad4$resample$Rsquared))
error_data8_quad <- add_row(error_data8_quad, r_sq = summary(data8_quad5)$r.squared, adj_r_sq = summary(data8_quad5)$adj.r.squared, cv_r_sq = mean(cv_data8_quad5$resample$Rsquared))
error_data8_quad <- add_row(error_data8_quad, r_sq = summary(data8_quad6)$r.squared, adj_r_sq = summary(data8_quad6)$adj.r.squared, cv_r_sq = mean(cv_data8_quad6$resample$Rsquared))
error_data8_quad <- add_row(error_data8_quad, r_sq = summary(data8_quad7)$r.squared, adj_r_sq = summary(data8_quad7)$adj.r.squared, cv_r_sq = mean(cv_data8_quad7$resample$Rsquared))


#plot of r square, adjusted r square and r square cross-validation
plot(error_data8_quad$r_sq, type = 'l', col = 'red', main = "ERROR PLOT QUAD", ylab = "Errors",xlab = "num variable", ylim = c(0,1))
lines(error_data8_quad$adj_r_sq,  col = 'green' )
lines(error_data8_quad$cv_r_sq,  col = 'blue')
legend(1,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"),
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)


## RESPONSE

#creating a new dataframe from auto df for RESPONSE SURFACE REGRESSION
data8 <- data8

data8_rspns1 <- rsm(id_android ~ SO(rating_weather, car_or_bus), data = data8)
data8_rspns2 <- rsm(id_android ~ SO(rating_weather, car_or_bus, speed), data = data8)
data8_rspns3 <- rsm(id_android ~ SO(rating_weather, car_or_bus, speed, time), data = data8)
data8_rspns4 <- rsm(id_android ~ SO(rating_weather, car_or_bus, speed, time, rating_bus), data = data8)
data8_rspns5 <- rsm(id_android ~ SO(rating_weather, car_or_bus, speed, time, rating_bus, distance), data = data8)
data8_rspns6 <- rsm(id_android ~ SO(rating_weather, car_or_bus, speed, time, rating_bus, distance, rating), data = data8)



error_data8_rspns <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0))
error_data8_rspns <- add_row(error_data8_rspns, r_sq = summary(data8_rspns1)$r.squared, adj_r_sq = summary(data8_rspns1)$adj.r.squared)
error_data8_rspns <- add_row(error_data8_rspns, r_sq = summary(data8_rspns2)$r.squared, adj_r_sq = summary(data8_rspns2)$adj.r.squared)
error_data8_rspns <- add_row(error_data8_rspns, r_sq = summary(data8_rspns3)$r.squared, adj_r_sq = summary(data8_rspns3)$adj.r.squared)
error_data8_rspns <- add_row(error_data8_rspns, r_sq = summary(data8_rspns4)$r.squared, adj_r_sq = summary(data8_rspns4)$adj.r.squared)
error_data8_rspns <- add_row(error_data8_rspns, r_sq = summary(data8_rspns5)$r.squared, adj_r_sq = summary(data8_rspns5)$adj.r.squared)
error_data8_rspns <- add_row(error_data8_rspns, r_sq = summary(data8_rspns6)$r.squared, adj_r_sq = summary(data8_rspns6)$adj.r.squared)


#plot of r square, adjusted r square
plot(error_data8_rspns$r_sq, type = 'l', col = 'red', main = "ERROR PLOT RESPONSE SURFACE", ylab = "Errors", ylim = c(0.3,1))
lines(error_data8_rspns$adj_r_sq,  col = 'green' )
legend(5,0.8, legend = c("R Squared","Adj R Squared"),
       col = c("red","green"), lty = 1:2, cex = 0.8)





