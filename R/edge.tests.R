#' This function does the tests by Chen and Friedman
#' @param x  a matrix of numbers 
#' @param y a matrix of numbers 
#' @return a list with statistics and p values
edge.tests=function(x,y) {
   nx=nrow(x)
   ny=nrow(y)
   dxy=as.matrix(stats::dist(rbind(x,y)))
   counts=matrix(1, nx+ny,2)
   tmp=gTests::g.tests(gTests::getGraph(counts,dxy, 1),1:nx, (nx+1):(nx+ny))
   out=list(statistics=unlist(tmp)[c(1,3,5,7)],
            p.values=unlist(tmp)[c(2,4,6,8)])
   names(out$statistics)=paste0("CF",1:4)
   names(out$p.values)=paste0("CF",1:4)
   out
}
