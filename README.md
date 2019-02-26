# Data_Science2
# About Project:
Implement different regression Technique such as: 
* Linear Regression
* Ridge Regression
* Lasso Regression
* Quad Regression
* Reponse Surface

Implement forward selection and cross validation on these regression technique and observe differnce in R square 
adjusted R square and cross validated R square using graph.

# How to run:
Scalation:
* Go inside Regression folder using terminal
* Run sbt server inisde the Regression folder:     $ sbt
* To compile:     sbt:Regression > compile
* To run:         sbt:Regression > run

Using built in dataset: Unzip the folder then follow the above instruction mentioned step. After run you will have the menu on which dataset you want to run the code. Once you make your choice the code will give you the graph for all the five regression technique mentined above.The graph has three value r^2, adjusted r^2 and cross validated r^2 after applying forward selection.    

For other dataset:
The target variable should be the first column of the dataset provided.
Path of the dataset should be provide properly to run the code.


R :
* Copy and paste the dataset's .csv files on your local directory.
* cd to the directory you have pasted all the files to.
* Open R Studio or go to the R shell by $R from the same directory.
* Copy and paste the R script in the submission.


# DataSet used: 

#### Data set used for R and Scalation both:
* Auto-Mpg
* Computer Hardware
* Concrete
* Graduate Admission
* Real Estate
* Yacht
* Gps Trajectories

#### Dataset used only on R:
* Airfoil
* Crane Controller

#### Dataset used only on Scalation:
* Beijing PM
* Red wine
* White wine
 
Note: All the data set has been taken from http://archive.ics.uci.edu/ml/datasets.html

# Technology Used:
* R
* Scala
* Scalation
* sbt

# Team Member:
* Priyank Malviya(Priyank.Malviya@uga.edu) 
* Anubhav Nigam (Anubhav.Nigam@uga.edu)
* Anant Tripathi (Anant.Tripathi@uga.edu)

# License
This project is licensed under the MIT License.


