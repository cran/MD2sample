#' This function does the chi square two sample test for continuous data in two dimensions
#' @param dta a list with values and counts 
#' @param minexpcount =5 minimum counts required per bin
#' @return a list with statistics and p values
chisq2D_test_disc=function(dta, minexpcount=5) {
  if(is.list(dta))
     dta=cbind(dta[["vals_x"]], dta[["vals_y"]], dta[["x"]], dta[["y"]])
  colnames(dta)=c("vals_x", "vals_y", "x", "y")
  tmp=0
  vx=sort(unique(dta[,1]))
  vy=sort(unique(dta[,2]))
  nbins=c(length(vx),length(vy))
  out=list(statistics=tmp, p.values=tmp, df=tmp)
  O_xy=matrix(0, nbins[1], nbins[2])
  dimnames(O_xy)=list(vx, vy)
  O_x=O_xy
  O_y=O_xy
  for(i in seq_along(vx)) {
    for(j in seq_along(vy)) {
      a=dta[((dta[,1]==vx[i])&(dta[,2]==vy[j])), ]
      O_x[i,j]=a[3]
      O_y[i,j]=a[4]
      O_xy[i,j]=O_x[i,j]+O_y[i,j]
    }
  }
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
       tmp=max(O_xy, na.rm = TRUE)
       nearest=c(-1,-1)
       for(step in 1:max(nbins)) {
         I=lowest[1]+c(-step:step)
         I=I[1<=I&I<=nbins[1]]
         J=lowest[2]+c(-step:step)
         J=J[1<=J&J<=nbins[2]]
         for(i in I) {
           for(j in J) {
             if(i==lowest[1]&&j==lowest[2]) next
             if(!is.na(O_xy[i,j])&&O_xy[i,j]<tmp) {
               tmp=O_xy[i,j]
               nearest = c(i,j)
             }
           }
         }
         if(nearest[1]>0) break
       }
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
  names(chi)="ChiSquare"
  df=length(O_x)-1
  names(df)="ChiSquare"
  out[[1]]=chi
  p=1-stats::pchisq(chi,df)
  names(p)="ChiSquare"  
  out[[2]]=p
  out[[3]]=df
  out
}
