#' Count Events in Bins
#' 
#' This function discretizes continuous data in two dimensions
#' 
#' @param x a matrix of numbers or a list 
#' @param y a matrix of numbers 
#' @param nbins =c(5,5) number of bins in x and in y direction
#' @param  Ranges =matrix(c(-Inf, Inf, -Inf, Inf),2,2) a 2x2 matrix with lower and upper bounds
#' @keywords internal
#' @return a list counts and vals
#' @export
bincounterR=function(x, y, nbins=c(5, 5), Ranges =matrix(c(-Inf, Inf, -Inf, Inf),2,2)) {
  if(length(nbins)==1) nbins=c(nbins, nbins)
  if(is.list(x)) {
     dta=x
     x=dta$x
     y=dta$y
  }
  else dta=list(x=x, y=y)
  if(ncol(x)!=2) {message("This is for two dimensional data only!");return(NULL)}
  xy=rbind(x, y)
  low=Ranges[1, ]
  if(!is.finite(low[1])) low[1]=min(xy[,1])-1e-5
  if(!is.finite(low[2])) low[2]=min(xy[,2])-1e-5
  high=Ranges[2, ]
  if(!is.finite(high[1])) high[1]=max(xy[,1])+1e-5
  if(!is.finite(high[2])) high[2]=max(xy[,2])+1e-5
  grd=as.list(1:2)
  for(i in 1:2) { # equal space grid
      grd[[i]]=seq(low[i], high[i], length=nbins[i]+1)
  }  
  midpoints_x=round((grd[[1]][-1]+grd[[1]][-(nbins[1]+1)])/2,2)
  midpoints_y=round((grd[[2]][-1]+grd[[2]][-(nbins[2]+1)])/2,2)
  vals=cbind(rep(midpoints_x, nbins[2]),rep(midpoints_y, each=nbins[1]))
  x=rbind(x, vals)
  y=rbind(y, vals)
  C_x=0*x
  C_y=0*y
  for(i in 1:2) {
    C_x[,i]=cut(x[,i], grd[[i]], labels=1:nbins[i],include.lowest =TRUE)
    C_y[,i]=cut(y[,i], grd[[i]], labels=1:nbins[i],include.lowest =TRUE)
  }
  tmp_x=table(C_x[,1],C_x[,2])-1
  tmp_y=table(C_y[,1],C_y[,2])-1  
  out=matrix(0, prod(nbins), 4)
  out[, 1:2]=vals
  for(i in 1:nbins[2]) {
     out[(i-1)*nbins[1]+1:nbins[1] , 3]=tmp_x[ ,i]
     out[(i-1)*nbins[1]+1:nbins[1] , 4]=tmp_y[ ,i]
  }
  colnames(out)=c("vals_x", "vals_y", "x", "y")
  out[order(out[,1], out[,2]), ]
}
