#' test function
#' @param  dta data set
#' @param  TS test statistics
#' @param  typeTS format of TS
#' @param  TSextra additional info TS
#' @return Mean computation time
#' @export
timecheck=function(dta, TS, typeTS, TSextra) {
  f=function() calcTS(dta, TS, typeTS, TSextra)
  a=microbenchmark::microbenchmark(f(), 
              unit="seconds", times=10)
  out=c(as.numeric(summary(a)["mean"]), 0)
  if(typeTS==1 & !("rnull"%in%names(TSextra))) {
    f=function() TS_cont_pval(dta$x, dta$y)
    a=microbenchmark::microbenchmark(f(), 
              unit="seconds", times=5)
    out[2]=as.numeric(summary(a)["mean"])
  }
  out
}
