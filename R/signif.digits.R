#' This function does some rounding to nice numbers
#' @param  x a list of two vectors
#' @param  d =4 number of digits to round to
#' @return A list with rounded vectors
signif.digits=function(x, d=4) {
   stat=x$statistics
   pval=x$p.values
   pval=round(pval, 4)
   y=strsplit(as.character(stat), "\\.")
   z=rep(0, length(y))
   for(i in 1:length(y)) {
      m=nchar(y[[i]][1])
      if(m>d) z[i]=round(stat[i], d-m)
      if(m<d & floor(stat[i])!=0) z[i]=round(stat[i], d-m+1)
      if(floor(stat[i])==0) {
        k=0
        repeat {
          k=k+1
          if(abs(stat[i]<1e-6)) break
          if(k>nchar(y[[i]][2])) break
          if(as.numeric(substr(y[[i]][2], 1, k))!=0) break
        }
        if(abs(stat[i]<1e-10)) tmp="0"
        else tmp=paste0("0.", paste0(rep("0",k-1),collapse=""), substr(y[[i]][2],k,d+k-1), collapse = "")
        z[i]=as.numeric(tmp)   
      } 
   }
   names(z)=names(pval)
   list(statistics=z, p.values=pval)
}
