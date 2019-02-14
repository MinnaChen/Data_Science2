library(ggpubr)
library(ggplot2)
library(e1071)
library(dpylr)

df <- read.csv('/home/mr_malviya/Desktop/Data_Science_2/ls')

#checking the structure of the df
str(df)

#making a copy of the original dataframe
auto_df <- df

#replacing the missing values by mean
auto_df$cylinders[is.na(auto_df$cylinders)] <- mean(auto_df$cylinders, na.rm = TRUE)
auto_df$displacement[is.na(auto_df$displacement)] <- mean(auto_df$displacement, na.rm = TRUE)
auto_df$weight[is.na(auto_df$weight)] <- mean(auto_df$weight, na.rm = TRUE)
auto_df$acceleration[is.na(auto_df$acceleration)] <- mean(auto_df$acceleration, na.rm = TRUE)
auto_df$model.year[is.na(auto_df$model.year)] <- mean(auto_df$model.year, na.rm = TRUE)
auto_df$origin[is.na(auto_df$origin)] <- mean(auto_df$origin, na.rm = TRUE)

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
fwd_model <- lm(mpg ~ 1,  data = auto_df) 
step(fwd_model, direction = "forward", scope = formula(mpg ~ cylinders + displacement + horsepower + weight + acceleration + model.year + origin))
summary(fwd_model)
## features obtained from forward selection mpg ~ weight + model.year + horsepower + origin + acceleration

#creating model with just cylinder
lin_mod1 <- lm(mpg ~ weight, data = auto_df)
lin_mod2 <- lm(mpg ~ weight + model.year, data = auto_df)
lin_mod3 <- lm(mpg ~ weight + model.year + horsepower, data = auto_df)
lin_mod4 <- lm(mpg ~ weight + model.year + horsepower + origin, data = auto_df)
lin_mod5 <- lm(mpg ~ weight + model.year + horsepower + origin + acceleration, data = auto_df)
lin_mod6 <- lm(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders, data = auto_df)
lin_mod7 <- lm(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders + displacement, data = auto_df)


#cross-validation from here
train_control <- trainControl(method = "cv", number = 10)
cv_model1 <- train(mpg ~ weight, data = auto_df, trControl = train_control, method = "lm")
cv_model2 <- train(mpg ~ weight + model.year, data = auto_df, trControl = train_control, method = "lm")
cv_model3 <- train(mpg ~ weight + model.year + horsepower, data = auto_df, trControl = train_control, method = "lm")
cv_model4 <- train(mpg ~ weight + model.year + horsepower + origin, data = auto_df, trControl = train_control, method = "lm")
cv_model5 <- train(mpg ~ weight + model.year + horsepower + origin + acceleration, data = auto_df, trControl = train_control, method = "lm")
cv_model6 <- train(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders, data = auto_df, trControl = train_control, method = "lm")
cv_model7 <- train(mpg ~ weight + model.year + horsepower + origin + acceleration + cylinders + displacement, data = auto_df, trControl = train_control, method = "lm")

                  
#creating a new dataframe to store rsquare adjusted_rsquard and cv rsquared
error_df <- data.frame("r_sq" = double(0), "adj_r_sq" = double(0), "cv_r_sq" = double(0))
error_df <- add_row(error_df, r_sq = summary(lin_mod1)$r.squared, adj_r_sq = summary(lin_mod1)$adj.r.squared, cv_r_sq = mean(cv_model1$resample$Rsquared))
error_df <- add_row(error_df, r_sq = summary(lin_mod2)$r.squared, adj_r_sq = summary(lin_mod2)$adj.r.squared, cv_r_sq = mean(cv_model2$resample$Rsquared))
error_df <- add_row(error_df, r_sq = summary(lin_mod3)$r.squared, adj_r_sq = summary(lin_mod3)$adj.r.squared, cv_r_sq = mean(cv_model3$resample$Rsquared))
error_df <- add_row(error_df, r_sq = summary(lin_mod4)$r.squared, adj_r_sq = summary(lin_mod4)$adj.r.squared, cv_r_sq = mean(cv_model4$resample$Rsquared))
error_df <- add_row(error_df, r_sq = summary(lin_mod5)$r.squared, adj_r_sq = summary(lin_mod5)$adj.r.squared, cv_r_sq = mean(cv_model5$resample$Rsquared))
error_df <- add_row(error_df, r_sq = summary(lin_mod6)$r.squared, adj_r_sq = summary(lin_mod6)$adj.r.squared, cv_r_sq = mean(cv_model6$resample$Rsquared))
error_df <- add_row(error_df, r_sq = summary(lin_mod7)$r.squared, adj_r_sq = summary(lin_mod7)$adj.r.squared, cv_r_sq = mean(cv_model7$resample$Rsquared))


#plot of r square, adjusted r square and r square cross-validation
plot(error_df$r_sq, type = 'l', col = 'red', main = "ERROR PLOT", ylab = "Errors", ylim = c(0.7,1))
lines(error_df$adj_r_sq,  col = 'green' )
lines(error_df$cv_r_sq,  col = 'blue')
legend(5,0.8, legend = c("R Squared","Adj R Squared","R Squared CV"), 
       col = c("red","green","blue"), lty = 1:2, cex = 0.8)






