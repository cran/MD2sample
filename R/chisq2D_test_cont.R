#' This function does the chi square two sample test for continuous data in two dimensions
#' @param dta_x  a matrix of numbers 
#' @param dta_y a matrix of numbers 
#' @param Ranges =matrix(c(-Inf, Inf, -Inf, Inf),2,2) a 2x2 matrix with lower and upper bounds
#' @param nbins =c(5,5) number of bins in x and y direction
#' @param minexpcount =5 minimum counts required per bin
#' @return a list with statistics and p values
chisq2D_test_cont=function(dta_x, dta_y, Ranges =matrix(c(-Inf, Inf, -Inf, Inf),2,2),
                           nbins=c(5, 5), minexpcount=5) {
  if(ncol(dta_x)!=2) {message("Test is for two dimensional data only!");return(NULL)}
  dta_xy=rbind(dta_x, dta_y)
  low=low=apply(dta_xy,2,min)
  if(is.finite(Ranges[1,1])) low[1]=Ranges[1,1]
  if(is.finite(Ranges[1,2])) low[2]=Ranges[1,2]
  high=apply(dta_xy,2,max)
  if(is.finite(Ranges[2,1])) high[1]=Ranges[2,1]
  if(is.finite(Ranges[2,2])) high[2]=Ranges[2,2]
  tmp=c(0,0)
  names(tmp)=c("ES", "EP")
  out=list(statistics=tmp, p.values=tmp, df=tmp)
  for(l in 1:2) {
    if(l==1) {
      grd=as.list(1:2)
      for(i in 1:2) { # equal space grid
        grd[[i]]=seq(low[i], high[i], length=nbins[i]+1)
      }      
    }
    else {
      grd=as.list(1:2)
      for(i in 1:2) { #equal probability grid
        grd[[i]]=quantile(dta_xy[,i], 0:nbins[i]/nbins[i])
        grd[[i]][c(1,nbins[i]+1)]=c(0,1)
      }
    }  
    A_x=0*dta_x
    A_y=0*dta_y
    A_xy=0*dta_xy
    for(i in 1:2) {
       A_x[,i]=cut(dta_x[,i], grd[[i]], labels=1:nbins[i],include.lowest =TRUE)
       A_y[,i]=cut(dta_y[,i], grd[[i]], labels=1:nbins[i],include.lowest =TRUE)
       A_xy[,i]=cut(dta_xy[,i], grd[[i]], labels=1:nbins[i],include.lowest =TRUE)
    }
    O_xy=matrix(0, nbins[1], nbins[2])
    rownames(O_xy)=1:nbins[1]
    colnames(O_xy)=1:nbins[2]
    tmp=table(A_xy[,1],A_xy[,2])
    O_xy[rownames(tmp),colnames(tmp)]=tmp
    O_x=0*O_xy
    tmp=table(A_x[,1],A_x[,2])
    O_x[rownames(tmp),colnames(tmp)]=tmp
    O_y=0*O_xy
    tmp=table(A_y[,1],A_y[,2])
    O_y[rownames(tmp),colnames(tmp)]=tmp
    step=1
    repeat {
       if(min(O_xy, na.rm=TRUE)>=minexpcount) break
       lowest = c(1, 1)
       tmp = max(O_xy, na.rm = TRUE)
       for(i in 1:nbins[1]) {
         for(j in 1:nbins[2]) {
           if(!is.na(O_xy[i,j]) && O_xy[i,j]<tmp) {
             tmp=O_xy[i,j]
             lowest=c(i,j)
           }
         }
       }
       I=lowest[1]+c(-1,0,1)*step
       I=I[1<=I&I<=nbins[1]]
       J=lowest[2]+c(-1,0,1)*step
       J=J[1<=J&J<=nbins[2]]
       tmp=max(O_xy, na.rm = TRUE)
       nearest=c(-1,-1)
       for(i in I)
         for(j in J) {
           if(i==lowest[1]&&j==lowest[2]) next
           if(!is.na(O_xy[i,j])&&O_xy[i,j]<tmp) {
             tmp=O_xy[i,j]
             nearest = c(i,j)
           }
         }
       if(step>min(nbins)) {
         message("Not enough data for the number of bins.")
         message(paste0("Will try nbins=(",nbins[1]-1,",",nbins[2]-1,")"))
         return(chisq2D_test_cont(dta_x, dta_y, nbins-1, minexpcount))
       }   
       if(nearest[1]<0) {step=step+1;next}
       
       O_x[lowest[1], lowest[2]] = O_x[lowest[1], lowest[2]] + 
                                   O_x[nearest[1], nearest[2]]
       O_x[nearest[1], nearest[2]] = NA
       O_y[lowest[1], lowest[2]] = O_y[lowest[1], lowest[2]] + 
         O_y[nearest[1], nearest[2]]
       O_y[nearest[1], nearest[2]] = NA
       O_xy[lowest[1], lowest[2]] = O_xy[lowest[1], lowest[2]] + 
         O_xy[nearest[1], nearest[2]]
       O_xy[nearest[1], nearest[2]] = NA
       step=1
    }
    O_x=O_x[!is.na(O_x)]
    O_y=O_y[!is.na(O_y)]
    O_xy=O_xy[!is.na(O_xy)]
    s=sqrt(sum(O_x)/sum(O_y))
    chi=sum((O_x/s-s*O_y)^2/O_xy)
    df=length(O_x)-1
    out[[1]][l]=chi
    out[[2]][l]=1-stats::pchisq(chi,df)
    out[[3]][l]=df
  }  
  out
}
