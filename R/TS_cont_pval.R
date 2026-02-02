#' This function runs a number of two sample tests which find their own p value.
#' @param  x  a matrix of numbers if data is continuous or of counts  if data is discrete.
#' @param  y a matrix of numbers if data is continuous or of counts  if data is discrete.
#' @return A list of two numeric vectors, the test statistics and the p values. 
#' @export 
TS_cont_pval = function(x, y) {
   D=ncol(x)
   nx=nrow(x)
   ny=nrow(y)
   n=nx+ny
   # test by 
   out=list(1:7, 1:7)
   names(out)=c("statistics", "p.values")
   names(out[[1]])=c("FR", "NN0", paste0("CF",1:4), "Ball")
   names(out[[2]])=c("FR", "NN0", paste0("CF",1:4), "Ball")   
   
   # test by Friedman and Rafski
   tmp=try(FR.test(x,y),TRUE)
   if(!is.list(tmp)) tmp=list(statistic=-99, p.value=-99)
   out[[1]][1]=tmp[[1]]
   out[[2]][1]=tmp[[2]] 

#  Nearest Neighbor variant   
   NN = c(FNN::get.knn(rbind(x,y), 1)$nn.index)
   out[[1]][2]=sum(NN[1:nx]<=nx)/n
   out[[2]][2]=1-stats::pbinom(n*out[[1]][2], nx, nx/n)
#  four tests by Chen and Friedman
   tmp=try(edge.tests(x, y), TRUE)
   if(!is.list(tmp)) tmp=list(statistic=-99, p.value=-99)
   out[[1]][3:6]=tmp[[1]]
   out[[2]][3:6]=tmp[[2]]
   tmp=try(Ball::bd.test(x, y, method="limit"), TRUE)
   if(!is.list(tmp)) tmp=list(statistic=-99, p.value=-99)
   out[[1]][7]=tmp[["statistic"]]
   out[[2]][7]=tmp[["p.value"]]
#  return p values   

   out
}
