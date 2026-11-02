#' This function runs a number of two sample tests which find their own p value.
#' @param  x  a matrix of numbers if data is continuous or of counts  if data is discrete.
#' @param  y a matrix of numbers if data is continuous or of counts  if data is discrete.
#' @param  DoBall =TRUE, run the Ball divergence test? Its large sample p value needs 
#'         the eigenvalues of a matrix of the size of the combined data set, so it is 
#'         slow for large data sets. If FALSE its statistic and p value are NA.
#' @return A list of two numeric vectors, the test statistics and the p values. 
#' @export 
TS_cont_pval = function(x, y, DoBall=TRUE) {
   D=ncol(x)
   nx=nrow(x)
   ny=nrow(y)
   n=nx+ny
   # test by 
   out=list(1:7, 1:7)
   names(out)=c("statistics", "p.values")
   names(out[[1]])=c("FR", "NN0", paste0("CF",1:4), "Ball")
   names(out[[2]])=c("FR", "NN0", paste0("CF",1:4), "Ball")   
   
#  Nearest Neighbor test, exact permutation mean and variance (Schilling 1986)   
   NN = c(FNN::get.knn(rbind(x,y), 1)$nn.index)
   tmp=NN0_test(NN, nx)
   out[[1]][2]=tmp$statistic
   out[[2]][2]=tmp$p.value
#  four tests by Chen and Friedman: CF1 original, CF2 generalized, CF3 weighted
#  and CF4 max-type edge-count test (in the order gTests::g.tests returns them)
   tmp=try(edge.tests(x, y), TRUE)
   if(!is.list(tmp)) tmp=list(statistic=-99, p.value=-99)
   out[[1]][3:6]=tmp[[1]]
   out[[2]][3:6]=tmp[[2]]
#  The test by Friedman and Rafsky is the original edge-count test on the 
#  minimum spanning tree, so it is the same test as CF1 (FR.test() gives the
#  same statistic and p value on the same tree). It is kept under its own name 
#  for backward compatibility.
   out[[1]][1]=out[[1]][3]
   out[[2]][1]=out[[2]][3]
   if(isFALSE(DoBall)) tmp=list(statistic=NA_real_, p.value=NA_real_)
   else {
     tmp=try(Ball::bd.test(x, y, method="limit"), TRUE)
     if(!is.list(tmp)) tmp=list(statistic=-99, p.value=-99)
   }
   out[[1]][7]=tmp[["statistic"]]
   out[[2]][7]=tmp[["p.value"]]
#  return p values   

   out
}

#' Simple nearest neighbor test
#'
#' C is the number of points of the first sample whose nearest neighbor is 
#' also from the first sample. Under the permutation null hypothesis the exact 
#' mean and variance of C are found from the nearest neighbor graph, 
#' see Schilling (1986), and the p value uses a normal approximation with 
#' continuity correction. Large values of C lead to rejection.
#' @param nn integer vector, `nn[i]` is the index of the nearest neighbor of 
#'        point i in the combined sample, the first nx points are from sample 1.
#' @param nx size of the first sample
#' @return a list with the statistic C/n and the p value
#' @keywords internal
NN0_test=function(nn, nx) {
   n=length(nn)
   C=sum(nn[1:nx]<=nx)
   # probability that r specific points all belong to sample 1
   pr=function(r) prod((nx-0:(r-1))/(n-0:(r-1)))
   # E[C^2]=sum over pairs (i,k) of P(i,nn[i],k,nn[k] all in sample 1),
   # which depends only on the number of distinct points among them
   M=sum(nn[nn]==seq_len(n))        # ordered mutual neighbor pairs
   indeg=tabulate(nn, nbins=n)       # how often a point is a nearest neighbor
   S2=n+M
   S3=sum(indeg*(indeg-1))+2*(n-M)
   S4=n*n-S2-S3
   EC=n*pr(2)
   VC=S2*pr(2)+S3*pr(3)+S4*pr(4)-EC^2
   p=if(VC>0) 1-stats::pnorm((C-0.5-EC)/sqrt(VC)) else NA
   list(statistic=C/n, p.value=p)
}
