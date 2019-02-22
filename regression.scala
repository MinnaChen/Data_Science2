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


object regression extends App {
  
  def Regression(x: => MatriD, y: => VectorD ){
     banner("Regression")
     val len_col=x.dim2-1
     val rg = new Regression(x,y)
     val fcols = Set(0)
     val rsqaure=new VectorD(len_col)
     val arsqaure=new VectorD(len_col)
     val crsqaure=new VectorD(len_col)
     
     for (l <- 0 until len_col) {
     val (x_j, b_j, fit_j) = rg.forwardSel (fcols) // add most predictive variable
     fcols +=x_j
     val fitnew=new Regression(x.selectCols(fcols.toArray),y)
     val result=fitnew.crossVal()  
     crsqaure(l)=result(fitnew.index_rSq).mean
     println(result(rg.index_rSq).mean)
     rsqaure(l)=fit_j(0)
     arsqaure(l)=fit_j(7) 
    
    }
     println("crsqaure",crsqaure)
     println("arsqaure",arsqaure)
     println("rsqaure",rsqaure)
     
     val size=x.dim1
     val xg = VectorD.range(0, len_col )
     val plot_mat = new MatrixD (3, len_col)
		 plot_mat.update(0, rsqaure)
		 plot_mat.update(1, arsqaure)
		 plot_mat.update(2, crsqaure)
		 new PlotM(xg, plot_mat,Array("Rsquare","aRsquare","crRsquare"), lines=true).setTitle("Regression")
     
     
     //val plot=new Plot (xg,arsqaure,rsqaure,"Regression", lines = true)
   
  }
  
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
     
     for (l <- 0 until len_colr) {
     val (x_jr, b_jr, fit_jr) = rrg.forwardSel (fcolsrrg) // add most predictive variable
     fcolsrrg +=x_jr
     val fitnew=new RidgeRegression(x.selectCols(fcolsrrg.toArray),y)
     val result=fitnew.crossVal()  
     crsqaure(l)=result(fitnew.index_rSq).mean
     rsqaure(l)=fit_jr(0)
     arsqaure(l)=fit_jr(7)
    }
     val size=x.dim1
     val xg = VectorD.range(0, len_colr )
     val plot_mat = new MatrixD (3, len_colr)
		 plot_mat.update(0, rsqaure)
		 plot_mat.update(1, arsqaure)
		 plot_mat.update(2, crsqaure)
		 new PlotM(xg, plot_mat, Array("Rsquare","adujsted Rsquare","crRsquare"), lines=true).setTitle("Ridge Regression") 
  }
  
  def LassoRegression(x: => MatriD, y: => VectorD){
     
    banner("Lasso Regression")
     val qrg=new LassoRegression(x,y)
     val len_col=x.dim2-1
     val fcols = Set(0)
     val rsqaure=new VectorD(len_col)
     val arsqaure=new VectorD(len_col)
     val crsqaure=new VectorD(len_col)
     for (l <- 0 until len_col) {
     val (x_j, b_jr, fit_j) = qrg.forwardSel (fcols) // add most predictive variable
     fcols +=x_j
     val fitnew=new LassoRegression(x.selectCols(fcols.toArray),y)
     val result=fitnew.crossVal()  
     crsqaure(l)=result(fitnew.index_rSq).mean.toDouble
     rsqaure(l)=fit_j(0)
     arsqaure(l)=fit_j(7)
    }
     val size=x.dim1
     val xg = VectorD.range(0, len_col )
     val plot_mat = new MatrixD (3, len_col)
		 plot_mat.update(0, rsqaure)
		 plot_mat.update(1, arsqaure)
		 plot_mat.update(2, crsqaure)
		 new PlotM(xg, plot_mat,Array("Rsquare","adujsted Rsquare","crRsquare"), lines=true).setTitle("Lasso Regression")
    
  }
  
  def QuadRegression(x: => MatriD, y: => VectorD){
    
     banner("Quad Regression")
     val qrg=new QuadRegression(x,y)
     val x1 = qrg.getX
     val len_col=x.dim2
     val fcols = Set(0)
     val rsqaure=new VectorD(len_col)
     val arsqaure=new VectorD(len_col)
     val crsqaure=new VectorD(len_col)
     for (l <- 0 until len_col) {
     val (x_j, b_jr, fit_j) = qrg.forwardSel (fcols) // add most predictive variable 
     fcols +=x_j
     val fd=fcols.toArray
     val fitnew=new Regression(x1.selectCols(fd),y)
     val result=fitnew.crossVal()  
     
     crsqaure(l)=result(fitnew.index_rSq).mean
     rsqaure(l)=fit_j(0)
     arsqaure(l)=fit_j(7)
    }
     val size=x.dim1
     val xg = VectorD.range(0, len_col )
     val plot_mat = new MatrixD (3, len_col)
     plot_mat.update(0, rsqaure)
     plot_mat.update(1, arsqaure)
     plot_mat.update(2, crsqaure)
     new PlotM(xg, plot_mat,Array("Rsquare","adujsted Rsquare","crRsquare"), lines=true).setTitle("Quad Regression")
    
  }
  
  def ResponseSurface(x: => MatriD, y: => VectorD){
    
     banner("Response Surface")
     val rs=new ResponseSurface(x,y)
     val x1 = rs.getX
     val len_col=x.dim2-1
     val fcols = Set.empty[Int]
     val rsqaure=new VectorD(len_col)
     val arsqaure=new VectorD(len_col)
     val crsqaure=new VectorD(len_col)
     for (l <- 0 until len_col) {
     val (x_j, b_j, fit_j) = rs.forwardSel (fcols) // add most predictive variable
     fcols +=x_j
     val fitnew=new Regression(x1.selectCols(fcols.toArray),y)
     val result=fitnew.crossVal()  
     crsqaure(l)=result(fitnew.index_rSq).mean
     rsqaure(l)=fit_j(0)
     arsqaure(l)=fit_j(7)
     }
     val size=x.dim1
     val xg = VectorD.range(0, len_col )
     val plot_mat = new MatrixD (3, len_col)
		 plot_mat.update(0, rsqaure)
		 plot_mat.update(1, arsqaure)
		 plot_mat.update(2, crsqaure)
		 new PlotM(xg, plot_mat,Array("Rsquare","adujsted Rsquare","crRsquare"),lines=true).setTitle("Response Surface")
  }
  
  def main(){
    println("enter the name of data set")
    val filename=scala.io.StdIn.readLine()
    val fname= Array ("cylinders", "displacement", "horsepower", "weight", "acceleration",
                       "model_year", "origin", "mpg")
    //Converting data set to relations
     val relation= Relation("auto-mpg.csv", "relation",fname, -1 ,null)
     
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
     val (x,y)=relation.toMatriDD(1 until 8,0)
    Regression(x,y)
    RidgeRegression(x, y)
    LassoRegression(x, y)
    QuadRegression(x, y)
    ResponseSurface(x, y)       
  }
  main()
  
}
