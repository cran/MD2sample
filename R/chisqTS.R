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
    # assign each point to exactly one bin; the bins are closed on the left,
    # open on the right, except the last one which is closed on both sides.
    lev1=seq_len(nbins[k,1]); lev2=seq_len(nbins[k,2])
    binx1=factor(findInterval(x[,1], p1, rightmost.closed=TRUE, all.inside=TRUE), levels=lev1)
    binx2=factor(findInterval(x[,2], p2, rightmost.closed=TRUE, all.inside=TRUE), levels=lev2)
    biny1=factor(findInterval(y[,1], p1, rightmost.closed=TRUE, all.inside=TRUE), levels=lev1)
    biny2=factor(findInterval(y[,2], p2, rightmost.closed=TRUE, all.inside=TRUE), levels=lev2)
    Ox=unclass(table(binx1, binx2))
    Oy=unclass(table(biny1, biny2))
    Oxy=Ox+Oy
    I=(Oxy>3)    
    s=sqrt(sum(Ox[I])/sum(Oy[I]))
    chi[k]=sum((Ox[I]/s-s*Oy[I])^2/Oxy[I])
    df=length(Ox[I])-1
    pval[k]=1-stats::pchisq(chi[k],df)
    names(chi)[k]=paste0("Chisq Stat ", nbins[k,1], ",", nbins[k,2])
    names(pval)[k]=paste0("Chisq Pval ", nbins[k,1], ",", nbins[k,2])
  }  
  if(TSextra$which=="All") return(list(statistic=chi, p.value=pval, df=df))
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
