#' Helper function to find test statistics of simulated data.
#' @param  dta a list
#' @param TS test statistic routine
#' @param typeTS type of routine
#' @param TSextra a list
#' @param B number of simulation runs
#' @return a matrix
#' @keywords internal
#' @export
simTS=function(dta, TS, typeTS, TSextra, B) {
  A=matrix(0, B, length(calcTS(dta, TS, typeTS, TSextra)))
  for(i in 1:B) {
    simdta=gen_sim_data(dta, TSextra)
    if(typeTS==1) TSextra$distances=TSextra$dist(simdta)
    A[i,]=calcTS(simdta, TS, typeTS, TSextra)
  }
  A
}

#' Helper function to find test statistics of simulated data.
#' @param  dta a list
#' @param TS test statistic routine
#' @param typeTS type of routine
#' @param TSextra a list
#' @param A a matrix
#' @param Continuous logical
#' @param Ranges a matrix
#' @param nbins a vector
#' @param minexpcount an integer
#' @param B number of simulation runs
#' @return a matrix
#' @keywords internal
#' @export
simpvals=function(dta, TS, typeTS, TSextra, A, Continuous, 
                  Ranges, nbins, minexpcount, B) {
  num_tests=length(calcTS(dta, TS, typeTS, TSextra))
  pvalsTS=matrix(0, B, num_tests)
  if(Continuous) {
       pvalsChi=matrix(0, B, 2)
       colnames(pvalsChi)=c("ES","EP")
       tmp=TS_cont_pval(dta$x, dta$y)$p.values
       pvalsOther=matrix(0, B, length(tmp))
       colnames(pvalsOther)=names(tmp)
  }   
  else {
      pvalsChi=matrix(0, B, 1)
      colnames(pvalsChi)=c("Chisquare")
      pvalsOther=NULL
  }  
  for(i in 1:B) {
    simdta=gen_sim_data(dta, TSextra)
    if(typeTS==1) TSextra$distances=TSextra$dist(simdta)
    tmp=calcTS(simdta, TS, typeTS, TSextra)
    if(Continuous) {
        pvalsOther[i, ]=TS_cont_pval(simdta$x, simdta$y)$p.values 
        pvalsChi[i, ]=chisq2D_test_cont(simdta$x, simdta$y, Ranges, nbins, minexpcount)$p.values
    }
    else pvalsChi[i, ]=chisq2D_test_disc(simdta, minexpcount)$p.values
    for(j in 1:num_tests) pvalsTS[i,j]=pvalsTS[i,j]+sum(tmp[j]>A[,j])/nrow(A) 
  }
  colnames(pvalsTS)=names(tmp)
  list(pvalsTS=pvalsTS, pvalsOther=pvalsOther, pvalsChi=pvalsChi)
}

#' Adjusted p values
#' 
#' This function runs a number of two sample tests using Rcpp and parallel computing and then finds the correct p value for the combined tests.
#' 
#' For details consult the vignette("MD2sample","MD2sample")
#' 
#' @param  x  Continuous data: either a matrix of numbers, or a list with two matrices called x and y.
#'                             if it is a matrix Observations are in different rows.
#'            Discrete data: a vector of counts or a matrix with columns named vals_x, vals_y, x and y.
#' @param  y a matrix of numbers if data if data is continuous or a vector of counts  if data is discrete.
#' @param  vals_x =NA, a vector of values for discrete random variable, or NA if data is continuous.
#' @param  vals_y =NA, a vector of values for discrete random variable, or NA if data is continuous.
#' @param  B =c(5000, 1000), number of simulation runs for permutation test and for estimation
#'         of the empirical distribution function.
#' @param  nbins =c(5, 5), number of bins for chi square tests (2D only).
#' @param  minexpcount = 5, minimum required expected counts for chi-square tests.
#' @param  samplingmethod ="Binomial" or "independence" for discrete data.
#' @param  Ranges =matrix(c(-Inf, Inf, -Inf, Inf),2,2) a 2x2 matrix with lower and upper bounds.
#' @param  DoTransform =TRUE, should data be transformed to interval (0,1)?
#' @param  rnull routine for parametric bootstrap.
#' @param  SuppressMessages = FALSE, print informative messages?
#' @param  maxProcessor number of cores for parallel processing.
#' @param  doMethods  Which methods should be included? If missing a small number of methods that generally have good power are used.
#' @return NULL, results are printed out.
#' @examples
#' #Note that the number of simulation runs B is very small to
#' #satisfy CRAN's run time constraints. 
#' #Two continuous data sets from a multivariate normal:
#' x = mvtnorm::rmvnorm(100, c(0,0))
#' y = mvtnorm::rmvnorm(120, c(0,0))
#' twosample_test_adjusted_pvalue(x, y, maxProcessor=1, B=20)
#' #Two discrete data sets from some distribution:
#' x = table(sample(1:4, size=1000, replace = TRUE))
#' y = table(sample(1:4, size=500, replace = TRUE, prob=c(1, 1.5, 1, 1)))
#' twosample_test_adjusted_pvalue(x, y, rep(1:2,2), rep(1:2, each=2), maxProcessor=1, B=20)
#' @export
twosample_test_adjusted_pvalue=function(x, y, vals_x=NA, vals_y=NA,  
                        B=c(5000, 1000), nbins=c(5,5),
                        minexpcount=5, samplingmethod="Binomial",
                        Ranges =matrix(c(-Inf, Inf, -Inf, Inf),2,2),
                        DoTransform=TRUE, rnull, SuppressMessages=FALSE, 
                        maxProcessor, doMethods) {
    default.methods = list(cont=c("ES", "CvM", "AZ", "NN5", "BG"), 
                           disc=c("Chisquare", "KS", "AZ", "CvM"))
    all.methods = list(cont=c("KS", "K", "CvM","AD","NN1", "NN5", "AZ","BF",
                              "BG", "FR", "NN0", "CF1", "CF2", "CF3", "CF4",
                              "ES", "EP"),
                       disc=c("KS", "K", "CvM","AD","NN","AZ","BF","Chisquare"))                                          
    if(length(B)==1) B=c(B, B)
    if(!is.numeric(samplingmethod))
      samplingmethod=ifelse(samplingmethod=="independence", 1, 2)
    if(missing(y)) {
      if(is.list(x)) {#Continuous Data
        if(!SuppressMessages) message("Data is assumed to be continuous")
        Continuous=TRUE
        dta=x
        y=x$y
        x=x$x 
        Dim=ncol(x)
      }
      else {#Discrete data
        if(!SuppressMessages) message("Data is assumed to be discrete")
        Continuous=FALSE
        dta=x
        x=matrix(1:4,2,2)#just some dummy numbers
        y=matrix(1:4,2,2)
        dta=list(x=dta[,"x"], y=dta[,"y"], vals_x=dta[,"vals_x"], vals_y=dta[,"vals_y"])
        DoTransform=FALSE
      }
    }
    else {
      Continuous=ifelse(any(is.na(c(vals_x, vals_y))), TRUE, FALSE)
      if(Continuous) {
        if(!SuppressMessages) message("Data is assumed to be continuous")
        dta=list(x=x, y=y)
        Dim=ncol(x)
      } 
      else {
        if(!SuppressMessages) message("Data is assumed to be discrete")
        dta=list(x=x, y=y, vals_x=vals_x, vals_y=vals_y)
        x=matrix(1:4,2,2)#just some dummy numbers
        y=matrix(1:4,2,2)
        DoTransform=FALSE
      }    
    }
    if(Continuous) {
      if(nrow(y)<nrow(x)) { #switch x and y so that nx>ny
        tmp=y
        y=x
        x=tmp
        dta=list(x=x, y=y)
      }
      rawdta=dta
      if(DoTransform) {
          dta=transform01(dta)
          x=dta$x
          y=dta$y
          Ranges=matrix(c(0, 1, 0, 1),2,2)
      }
    }
    if(Continuous) 
         TSextra = list(knn=function(x) FNN::get.knn(x, 5)$nn.index,
                dist=function(dta) find_dist(dta),
                distances=find_dist(dta),
                DoTransform=DoTransform)
    else TSextra = list(
                dist=function(dta) NULL,
                organize=function(dta) dta=dta[order(dta[,1], dta[,2]), ],
                samplingmethod=samplingmethod)
    if(!missing(rnull)) {
      if(Continuous) 
        TSextra=c(TSextra, rnull=rnull, rawdta=list(rawdta))
      else TSextra=c(TSextra, rnull=rnull)
    } 
    outchi=list(statistics=NULL, p.value=NULL)
    outpvals=list(statistics=NULL, p.value=NULL)
    # what methods are to be run?  
    if(Continuous) {
        if(Dim==2) {
          if(length(nbins)==1) nbins=c(nbins, nbins)
          outchi = chisq2D_test_cont(x, y, Ranges, nbins, minexpcount)
        }  
        outpvals=TS_cont_pval(x, y) #Methods that find p values
        typeTS=1
        TS=TS_cont
        dta=list(x=x, y=y)
    }
    else {
        typeTS=4
        outchi = chisq2D_test_disc(dta,  minexpcount)
        TS=TS_disc
    }    
    TS_data=calcTS(dta, TS, typeTS, TSextra)
    if(any(is.null(names(TS_data)))) {
      if(!SuppressMessages) message("output of TS routine has to be a named vector!")
      return(NULL)
    }  
#   Find matrix of values of test statistics for simulated data
    if(missing(maxProcessor)) {
       maxProcessor=parallel::detectCores(logical = FALSE)-1
       if(!SuppressMessages) message(paste("Using ", maxProcessor," cores.."))
    }   
    if(maxProcessor==1) A=simTS(dta, TS, typeTS, TSextra, B[1])
    else {
       cl=parallel::makeCluster(maxProcessor)
       z=parallel::clusterCall(cl, simTS, dta, TS, typeTS, 
                               TSextra, round(B[1]/maxProcessor))
       A=z[[1]]
       for(i in 2:maxProcessor) A=rbind(A, z[[i]])
       B[1]=nrow(A)

    }
    num_tests=ncol(A)
    tmp=TS_data
    pvalsdta=rep(0, num_tests)
    for(j in 1:num_tests) pvalsdta[j]=pvalsdta[j]+sum(tmp[j]<A[,j])/nrow(A)    
    if(Continuous) {
        pvalsdta=c(pvalsdta, TS_cont_pval(x, y)$p.values) 
        if(length(nbins)==1) nbins=c(nbins, nbins)
        pvalsdta=c(pvalsdta, chisq2D_test_cont(x, y, Ranges, nbins, minexpcount)$p.values)
        names(pvalsdta)=all.methods$cont
    }   
    else {
        pvalsdta=c(pvalsdta, chisq2D_test_disc(dta, minexpcount)$p.values)
        names(pvalsdta)=all.methods$disc
    }
    if(maxProcessor==1) {
        tmp=simpvals(dta, TS, typeTS, TSextra, A, Continuous, 
                 Ranges, nbins, minexpcount, B[2])
        pvalsTS=tmp$pvalsTS
        pvalsOther=tmp$pvalsOther
        pvalsChi=tmp$pvalsChi
    }
    else {
      z=parallel::clusterCall(cl, simpvals, dta, TS, 
                  typeTS, TSextra, A, Continuous, 
                  Ranges, nbins, minexpcount, B[2]/maxProcessor)
      pvalsTS=z[[1]][[1]]
      pvalsOther=z[[1]][[2]]
      pvalsChi=z[[1]][[3]]
      for(i in 2:maxProcessor) {
        pvalsTS=rbind(pvalsTS, z[[i]][[1]])
        pvalsOther=rbind(pvalsOther, z[[i]][[2]])
        pvalsChi=rbind(pvalsChi, z[[i]][[3]])
      }  
      B[1]=nrow(pvalsTS)
      parallel::stopCluster(cl)
    }
    if(missing(doMethods)) {
      if(Continuous) doMethods=default.methods[["cont"]]
      else doMethods=default.methods[["disc"]]
    }
    if(doMethods[1]=="all"){
      if(Continuous) doMethods=all.methods[["cont"]]
      else doMethods=all.methods[["disc"]]
    }
    pvals=cbind(pvalsTS, pvalsOther, pvalsChi) 
    pvals=pvals[ ,doMethods,drop=FALSE]
    pvalsdta=pvalsdta[doMethods]
    minp_x=min(pvalsdta)
    minp_sim=apply(pvals[, ,drop=FALSE], 1, min)
    z=seq(0, 1, length=250)
    y=z
    for(i in 1:250) y[i]=sum(minp_sim<=z[i])/length(minp_sim)
    I=c(1:250)[z>minp_x][1]-1
    slope=(y[I+1]-y[I])/(z[I+1]-z[I])
    minp_adj=round(y[I]+slope*(minp_x-z[I]),4)
    message("p values of individual tests:")
    for(i in seq_along(pvalsdta)) 
        message(paste(names(pvalsdta)[i],": ", round(pvalsdta[i],4)))
    message(paste0("adjusted p value of combined tests: ", minp_adj))
    
}
