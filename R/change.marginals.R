#' Change Marginals of 2D Data
#' 
#' This function modifies the marginals of some data set
#' 
#' @param dta a list with two data sets
#' @param which method for modifying the data sets.
#' @param theta how much to modify
#' @keywords internal
#' @return a list of functions
#' @export
change.marginals=function(dta, which, theta) {
  if(which=="Exponential") {
    for(i in 2:ncol(dta$x)) {
      dta$x[, i] =  qexp(dta$x[, i], 1)
      dta$x[, i][dta$x[, i]>3]=3/dta$x[, i][dta$x[, i]>3]
      dta$y[, i] =  qexp(dta$y[, i], theta)
      dta$y[, i][dta$y[, i]>3]=3/dta$y[, i][dta$y[, i]>3]
    }
  }
  if(which=="Linear") {
    if(theta>0)
       for(i in 2:ncol(dta$y)) 
         dta$y[,i]=(theta-1+sqrt((1-theta)^2+4*theta*dta$y[,i]))/2/theta
  }
  if(which=="Normal") {
    for(i in 2:ncol(dta$x)) {
      dta$x[,i] =  qnorm(dta$x[, i])
      dta$y[,i] =  qnorm(dta$y[, i], 0, theta)
    }  
  }
  dta
}