library(ggpubr)
library(ggplot2)
library(e1071)

df <- read.csv('/home/mr_malviya/Desktop/Data_Science_2/ls')

## transferring cylinders data into new dataframe
cylinders <- df[c("cylinders")]
## replacing missing values with mean of the data
for (i in 1:ncol(cylinders)){
    cylinders[is.na(cylinders[,i]),i] <- mean(cylinders[,i], na.rm = TRUE)
}


## transferring displacement data into new dataframe
displacement <- df[c("displacement")]
## replacing missing values with mean of the data
for (i in 1:ncol(displacement)){
    displacement[is.na(displacement[,i]),i] <- mean(displacement[,i], na.rm = TRUE)
}


## transferring horsepower data into new dataframe
horsepower <- df[c("horsepower")]
## replacing missing values with mean of the data
for (i in 1:ncol(horsepower)){
    horsepower[is.na(horsepower[,i]),i] <- mean(horsepower[,i], na.rm = TRUE)
}

## transferring weight data into new dataframe
weight <- df[c("weight")]
## replacing missing values with mean of the data
for (i in 1:ncol(weight)){
    weight[is.na(weight[,i]),i] <- mean(weight[,i], na.rm = TRUE)
}


## transferring acceleration data into new dataframe
acceleration <- df[c("acceleration")]
## replacing missing values with mean of the data
for (i in 1:ncol(acceleration)){
    acceleration[is.na(acceleration[,i]),i] <- mean(acceleration[,i], na.rm = TRUE)
}


## transferring model year data into new dataframe
model_year <- df[c("model.year")]
## replacing missing values with mean of the data
for (i in 1:ncol(model_year)){
    model_year[is.na(model_year[,i]),i] <- mean(model_year[,i], na.rm = TRUE)
}


## transferring origin data into new dataframe
origin <- df[c("origin")]
## replacing missing values with mean of the data
for (i in 1:ncol(origin)){
    origin[is.na(origin[,i]),i] <- mean(origin[,i], na.rm = TRUE)
}


#dropping each column except for the mpg
df <- df[-c(2:9)]

##adding the new columns with the means in missing values to the dataframe
df <- cbind(df,cylinders, displacement, horsepower, weight, acceleration, model_year, origin)

##forward selection from here
fwd_model <- lm(mpg ~ 1,  data = df) 
step(fwd_model, direction = "forward", scope = formula(mpg ~ cylinders + displacement + horsepower + weight + acceleration + model.year + origin))


summary(fwd_model)


