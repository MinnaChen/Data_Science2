package Regression

import scalation.linalgebra._
import scalation.analytics._
import scalation.stat.StatVector._
import scalation.math.double_exp
import scalation.plot.Plot
import PredictorMat.{analyze, pullResponse_}
import scalation.random.Normal
import RegTechnique._
import scala.collection.mutable.Set
import scalation.util.banner
import scalation.columnar_db._
import scalation.linalgebra.MatriD
import MissingValues.replaceMissingValues
import scalation.plot.PlotM
import scala.io.StdIn.{readLine,readInt}


object FileReader extends App {
  
  //Regression
  def Regression(x: => MatriD, y: => VectorD ){
     banner("Regression")
     val len_col=x.dim2-1
     val rg = new Regression(x,y)
     val fcols = Set(0)
     val rsqaure=new VectorD(len_col)
     val arsqaure=new VectorD(len_col)
     val crsqaure=new VectorD(len_col)
     
     //Forward Selection 
     for (l <- 0 until len_col) {
     val (x_j, b_j, fit_j) = rg.forwardSel (fcols) // add most predictive variable
     fcols +=x_j
     val fitnew=new Regression(x.selectCols(fcols.toArray),y)
     //Cross Validation
     val result=fitnew.crossVal()  
     crsqaure(l)=result(fitnew.index_rSq).mean
     println(result(rg.index_rSq).mean)
     rsqaure(l)=fit_j(0)
     arsqaure(l)=fit_j(7) 
    
    }  
     //Plotting Graph
     val size=x.dim1
     val xg = VectorD.range(0, len_col )
     val plot_mat = new MatrixD (3, len_col)
		 plot_mat.update(0, rsqaure)
		 plot_mat.update(1, arsqaure)
		 plot_mat.update(2, crsqaure)
		 new PlotM(xg, plot_mat,Array("Rsquare","aRsquare","crRsquare"), lines=true).setTitle("Regression")  
  }
  
  //Ridge Regression
  def RidgeRegression(x: => MatriD, y: => VectorD){
    
     banner("Ridge Regression")
     val mu_x=x.mean
     val mu_y=y.mean
     val xc=x-mu_x
     val yc=y-mu_y
     val len_colr=xc.dim2-1
     val rrg= new RidgeRegression(xc,yc)
     val fcolsrrg = Set(0)
     val rsqaure=new VectorD(len_colr)
     val arsqaure=new VectorD(len_colr)
     val crsqaure=new VectorD(len_colr)
     
     //Forward Selection
     for (l <- 0 until len_colr) {
     val (x_jr, b_jr, fit_jr) = rrg.forwardSel (fcolsrrg) // add most predictive variable
     fcolsrrg +=x_jr
     //Cross Validation
     val fitnew=new RidgeRegression(x.selectCols(fcolsrrg.toArray),y)
     val result=fitnew.crossVal()  
     crsqaure(l)=result(fitnew.index_rSq).mean
     rsqaure(l)=fit_jr(0)
     arsqaure(l)=fit_jr(7)
    }
     
     //Plotting Graph
     val size=x.dim1
     val xg = VectorD.range(0, len_colr )
     val plot_mat = new MatrixD (3, len_colr)
		 plot_mat.update(0, rsqaure)
		 plot_mat.update(1, arsqaure)
		 plot_mat.update(2, crsqaure)
		 new PlotM(xg, plot_mat, Array("Rsquare","aRsquare","crRsquare"), lines=true).setTitle("Ridge Regression") 
  }
  
  //Lasso Regression
  def LassoRegression(x: => MatriD, y: => VectorD){
     
    banner("Lasso Regression")
     val qrg=new LassoRegression(x,y)
     val len_col=x.dim2-1
     val fcols = Set(0)
     val rsqaure=new VectorD(len_col)
     val arsqaure=new VectorD(len_col)
     val crsqaure=new VectorD(len_col)
    
     //Forward Selection
     for (l <- 0 until len_col) {
     val (x_j, b_jr, fit_j) = qrg.forwardSel (fcols) // add most predictive variable
     fcols +=x_j
     val fitnew=new LassoRegression(x.selectCols(fcols.toArray),y)
     //Cross Validation
     val result=fitnew.crossVal()  
     crsqaure(l)=result(fitnew.index_rSq).mean.toDouble
     rsqaure(l)=fit_j(0)
     arsqaure(l)=fit_j(7)
    }
    
    //Plotting Graph
     val size=x.dim1
     val xg = VectorD.range(0, len_col )
     val plot_mat = new MatrixD (3, len_col)
		 plot_mat.update(0, rsqaure)
		 plot_mat.update(1, arsqaure)
		 plot_mat.update(2, crsqaure)
		 new PlotM(xg, plot_mat,Array("Rsquare","aRsquare","crRsquare"), lines=true).setTitle("Lasso Regression")  
  }
  
  //Quad Regression
  def QuadRegression(x: => MatriD, y: => VectorD){
    
     banner("Quad Regression")
     val qrg=new QuadRegression(x,y)
     val x1 = qrg.getX
     val len_col=x.dim2
     val fcols = Set(0)
     val rsqaure=new VectorD(len_col)
     val arsqaure=new VectorD(len_col)
     val crsqaure=new VectorD(len_col)
     
     //Forward Selection
     for (l <- 0 until len_col) {
     val (x_j, b_jr, fit_j) = qrg.forwardSel (fcols) // add most predictive variable 
     fcols +=x_j
     val fd=fcols.toArray
     val fitnew=new Regression(x1.selectCols(fd),y)
     //Cross validation
     val result=fitnew.crossVal()   
     crsqaure(l)=result(fitnew.index_rSq).mean
     rsqaure(l)=fit_j(0)
     arsqaure(l)=fit_j(7)
    }
     
     //Plotting Graph
     val size=x.dim1
     val xg = VectorD.range(0, len_col )
     val plot_mat = new MatrixD (3, len_col)
     plot_mat.update(0, rsqaure)
     plot_mat.update(1, arsqaure)
     plot_mat.update(2, crsqaure)
     new PlotM(xg, plot_mat,Array("Rsquare","aRsquare","crRsquare"), lines=true).setTitle("Quad Regression")    
  }
  
  //Response Surface
  def ResponseSurface(x: => MatriD, y: => VectorD){
    
     banner("Response Surface")
     val rs=new ResponseSurface(x,y)
     val x1 = rs.getX
     val len_col=x.dim2-1
     val fcols = Set.empty[Int]
     val rsqaure=new VectorD(len_col)
     val arsqaure=new VectorD(len_col)
     val crsqaure=new VectorD(len_col)
     
     //Forward Selection
     for (l <- 0 until len_col) {
     val (x_j, b_j, fit_j) = rs.forwardSel (fcols) // add most predictive variable
     fcols +=x_j
     val fitnew=new Regression(x1.selectCols(fcols.toArray),y)
     //Cross Validation
     val result=fitnew.crossVal()  
     crsqaure(l)=result(fitnew.index_rSq).mean
     rsqaure(l)=fit_j(0)
     arsqaure(l)=fit_j(7)
     }
     
     //Plotting Graph
     val size=x.dim1
     val xg = VectorD.range(0, len_col )
     val plot_mat = new MatrixD (3, len_col)
		 plot_mat.update(0, rsqaure)
		 plot_mat.update(1, arsqaure)
		 plot_mat.update(2, crsqaure)
		 new PlotM(xg, plot_mat,Array("Rsquare","aRsquare","crRsquare"),lines=true).setTitle("Response Surface")
  }
  
  def  fileInformation = {
    val i = scala.io.StdIn.readLine().toInt
    val (fileName : String , fName : Array[String]) = i match {
    case 1  => ("auto-mpg.csv",
               Array("mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration",
                       "model_year", "origin"))
    case 2  => ("gps.csv",
               Array("id",	"id_android",	"speed",	"time",	"distance",	"rating",	"rating_bus",	"rating_weather",	"car_or_bus"))
    case 3  => ("real_estate.csv",
               Array("X1 transaction date","X2 house age","X3 distance to the nearest MRT station","X4 number of convenience stores","X5 latitude","X6 longitude","Y house price of unit area"))
    case 4  => ("Concrete_Data.csv",
               Array("Cement", "Blast furnace slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age", "Concrete compressive strength"))
    case 5  =>("Admission_Predict.csv",
               Array("GRE Score","TOEFL Score","University Rating","SOP","LOR","CGPA","Research","Chance of Admit "))
    case 6  =>("computerhardware.csv",
               Array("erp","myct","mmin","mmax","cach", "chmin","chmax","prp"))
    case 7  =>("beijing_pm.csv",
               Array("pm2.5","year","month","day","hour","DEWP","TEMP","PRES","lws","ls","lr"))           
    case 8  =>("wine.csv",
               Array("quality","fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"))
    case 9  =>("red_wine.csv",
               Array("quality","fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"))
    case 10 =>("yachtdata.csv",
               Array("Longitudnal_position"," Prismatic_coef"," Length_displacement"," Beam_draught"," length_beam"," froude_number"," residuary_resistance"))
    case default => println(" Wrong input. You entered " + default+ " You need to select from 1 - 10") 
                    
    }
    (fileName,fName)
  }
  
  def main(){
    banner("List of Data Set")
    banner("\n 1. auto-mpg \n 2.GPS Trajectories  \n 3. Real estate valuation data set \n 4. Concrete Data \n 5. Admission Predict \n 6. Computer Hardware \n 7. Beijing PM \n 8. White Wine \n 9. Red Wine \n 10. Yatch Data ")
    
    println("Enter the number of data set to be selected")
    val (fileName, fName) = fileInformation
    //Converting data set to relations
    try{
     val relation= Relation(fileName, "relation", fName, -1 ,null)
   
     val len_relation= relation.cols
     // implemented to  calculate the mean value for all the column and put it in the  missing position 
     // in the data set
     
     for(i <- 0 to len_relation-1){
       val replaced_value =relation.sigmaS(relation.colName(i), (value) => value!="?")
       // to extract each column from matrix xy
       var col_vec=replaced_value.col(i)
       col_vec=col_vec.asInstanceOf [VectorS].toDouble
       //calculated the mean value of the column
       val im=ImputeMean.impute(col_vec)
       relation.update(relation.colName(i), im.toString(), "?")
      }  
    
     val (x,y)=relation.toMatriDD(1 until len_relation,0)
    Regression(x,y)
    RidgeRegression(x, y)
    LassoRegression(x, y)
    QuadRegression(x, y)
    ResponseSurface(x, y)
    }
//    catch
//    {
//      case unknown => println("Got this unknown exception: " + unknown)
//    }
  }
  main()
  
}
