#' Chi-square test for continuous data, example of user supplied function
#' 
#' This function does the chi square two sample test for continuous data in two dimensions
#' 
#' @param x  a matrix of numbers 
#' @param y a matrix of numbers 
#' @param TSextra list with some info
#' @return either a test statistic or a p value
#' @export
chiTS.cont=function(x, y, TSextra) {
  nbins=TSextra$nbins
  if(!is.matrix(nbins)) nbins=rbind(nbins)
  chi=rep(0, nrow(nbins))
  pval=rep(0, nrow(nbins))
  for(k in 1:nrow(nbins)) {
    p1=quantile(c(x[,1],y[,1]), (0:nbins[k,1])/nbins[k,1])
    p2=quantile(c(x[,2],y[,2]), (0:nbins[k,2])/nbins[k,2])
    Ox=Oy=matrix(0, nbins[k,1], nbins[k,2])
    for(i in 1:nbins[k,1]) {
      x1=x[x[,1]>=p1[i] & x[,1]<=p1[i+1], ]  
      y1=y[y[,1]>=p1[i] & y[,1]<=p1[i+1], ]
      for(j in 1:nbins[k,2]) {
        x2=x1[x1[,2]>=p2[j] & x1[,2]<=p2[j+1], ,drop=FALSE]  
        y2=y1[y1[,2]>=p2[j] & y1[,2]<=p2[j+1], ,drop=FALSE]
        Ox[i,j]=nrow(x2)
        Oy[i,j]=nrow(y2)
      }
    }
    I=(Oy+Ox>0)
    Oxy=Ox+Oy
    s=sqrt(sum(Ox[I])/sum(Oy[I]))
    chi[k]=sum((Ox[I]/s-s*Oy[I])^2/Oxy[I])
    df=length(Ox[I])-1
    pval[k]=1-stats::pchisq(chi[k],df)
    names(chi)[k]=paste0("Chisq Stat ", nbins[k,1], ",", nbins[k,2])
    names(pval)[k]=paste0("Chisq Pval ", nbins[k,1], ",", nbins[k,2])
  }  
  if(startsWith(TSextra$which, "stat")) return(chi)
  pval
}

#' Chi-square test for discrete data, example of user supplied function
#' 
#' This function does the chi square two sample test for discrete data in two dimensions
#' 
#' @param x  a vector with counts 
#' @param y  a vector with counts 
#' @param vals_x  a vector with points 
#' @param vals_y  a vector with points 
#' @param TSextra list with some info
#' @return either a test statistic or a p value
#' @export
chiTS.disc=function(x, y, vals_x, vals_y, TSextra) {
  I=(x+y>0)
  s=sqrt(sum(x[I])/sum(y[I]))
  chi=sum((x[I]/s-s*y[I])^2/(x+y)[I])
  names(chi)="Chisq Stat"
  df=length(x[I])-1
  if(startsWith(TSextra$which, "stat")) return(chi)
  pval=1-stats::pchisq(chi,df)
  names(pval)="Chisq P"
  pval
}
