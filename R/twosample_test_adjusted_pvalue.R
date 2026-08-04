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
  Dim=ncol(dta$x)
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
        if(Dim==2) 
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
  
  # Default subset of methods used when doMethods is not supplied.
  default.methods = list(cont=c("ES", "CvM", "AZ", "NN5", "BG"), 
                         disc=c("Chisquare", "KS", "AZ", "CvM"))
  
  # Full set of available methods for continuous and discrete data.
  all.methods = list(cont=c("KS", "K", "CvM","AD","NN1", "NN5", "AZ","BF",
                            "BG", "FR", "NN0", "CF1", "CF2", "CF3", "CF4",
                            "ES", "EP"),
                     disc=c("KS", "K", "CvM","AD","NN","AZ","BF","Chisquare"))                                          
  
  # B[1] is used to simulate test statistics.
  # B[2] is used to simulate p-values/min-p distribution.
  if(length(B)==1) B=c(B, B)
  
  # Convert sampling method to internal numeric code.
  # independence = 1, Binomial = 2.
  if(!is.numeric(samplingmethod))
    samplingmethod=ifelse(samplingmethod=="independence", 1, 2)
  
  # Determine whether data are continuous or discrete,
  # and whether y was supplied separately or bundled inside x.
  if(missing(y)) {
    
    if(is.list(x)) {
      # Continuous data supplied as list(x=..., y=...).
      if(!SuppressMessages) message("Data is assumed to be continuous")
      Continuous=TRUE
      dta=x
      y=x$y
      x=x$x 
      Dim=ncol(x)
    }
    else {
      # Discrete data supplied as matrix with columns:
      # x, y, vals_x, vals_y.
      if(!SuppressMessages) message("Data is assumed to be discrete")
      Continuous=FALSE
      Dim=2
      dta=x
      
      # Dummy matrices used only for compatibility with later references.
      x=matrix(1:4,2,2)
      y=matrix(1:4,2,2)
      
      dta=list(
        x=dta[,"x"], 
        y=dta[,"y"], 
        vals_x=dta[,"vals_x"], 
        vals_y=dta[,"vals_y"]
      )
      
      # Discrete data are not transformed.
      DoTransform=FALSE
    }
  }
  else {
    
    # If vals_x or vals_y is NA, assume continuous data.
    # Otherwise assume discrete data.
    Continuous=ifelse(any(is.na(c(vals_x, vals_y))), TRUE, FALSE)
    
    if(Continuous) {
      if(!SuppressMessages) message("Data is assumed to be continuous")
      dta=list(x=x, y=y)
      Dim=ncol(x)
    } 
    else {
      if(!SuppressMessages) message("Data is assumed to be discrete")
      Dim=2
      dta=list(x=x, y=y, vals_x=vals_x, vals_y=vals_y)
      
      # Dummy matrices used only for compatibility with later references.
      x=matrix(1:4,2,2)
      y=matrix(1:4,2,2)
      
      DoTransform=FALSE
    }    
  }
  
  if(Continuous) {
    
    # Ensure x is the smaller sample and y is the larger sample.
    if(nrow(y)<nrow(x)) {
      tmp=y
      y=x
      x=tmp
      dta=list(x=x, y=y)
    }
    
    # Save original data before optional transformation.
    rawdta=dta
    
    # Transform continuous data to the unit hypercube if requested.
    if(DoTransform) {
      dta=transform01(dta)
      x=dta$x
      y=dta$y
      Ranges=matrix(c(0, 1, 0, 1),2,2)
    }
  }
  
  # Create helper objects/functions needed by the test statistic routines.
  if(Continuous) 
    TSextra = list(
      knn=function(x) FNN::get.knn(x, 5)$nn.index,
      dist=function(dta) find_dist(dta),
      distances=find_dist(dta),
      DoTransform=DoTransform)
  else 
    TSextra = list(
      dist=function(dta) NULL,
      organize=function(dta) dta=dta[order(dta[,1], dta[,2]), ],
      samplingmethod=samplingmethod)
  
  # If a null-data generator is supplied, use it in the simulation routines.
  if(!missing(rnull)) {
    if(Continuous) 
      TSextra=c(TSextra, rnull=rnull, rawdta=list(rawdta))
    else 
      TSextra=c(TSextra, rnull=rnull)
  } 
  
  # Initialize objects for chi-square and analytic p-value results.
  outchi=list(statistics=NULL, p.value=NULL)
  outpvals=list(statistics=NULL, p.value=NULL)
  
  # Select built-in test-statistic routine.
  if(Continuous) {
    
    # Add chi-square tests for 2D continuous data.
    if(Dim==2) {
      if(length(nbins)==1) nbins=c(nbins, nbins)
      outchi = chisq2D_test_cont(x, y, Ranges, nbins, minexpcount)
    }  
    
    # Compute p-values for methods with direct p-value calculations.
    outpvals=TS_cont_pval(x, y)
    
    typeTS=1
    TS=TS_cont
    dta=list(x=x, y=y)
  }
  else {
    # Built-in discrete-data tests.
    typeTS=4
    outchi = chisq2D_test_disc(dta,  minexpcount)
    TS=TS_disc
  }    
  
  # Compute observed test statistics.
  TS_data=calcTS(dta, TS, typeTS, TSextra)
  
  # Require named output so methods can be matched correctly.
  if(any(is.null(names(TS_data)))) {
    if(!SuppressMessages) message("output of TS routine has to be a named vector!")
    return(NULL)
  }  
  
  # Simulate null distribution of test statistics.
  if(missing(maxProcessor)) {
    maxProcessor = max(parallel::detectCores(logical = FALSE)-1, 1)
    if(!SuppressMessages) message(paste("Using ", maxProcessor," cores.."))
  }   
  
  if(maxProcessor==1) {
    A=simTS(dta, TS, typeTS, TSextra, B[1])
  }
  else {
    # Parallel simulation of test statistics.
    cl1=parallel::makeCluster(maxProcessor)
    on.exit(parallel::stopCluster(cl1), add=TRUE)
    z=parallel::clusterCall(cl1, simTS, dta, TS, typeTS, 
                            TSextra, round(B[1]/maxProcessor))
    A=z[[1]]
    for(i in 2:maxProcessor) A=rbind(A, z[[i]])
    
    # Update B[1] to actual number of simulated rows.
    B[1]=nrow(A)
  }
  
  # Compute individual p-values from simulated null statistics.
  num_tests=ncol(A)
  tmp=TS_data
  pvalsdta=rep(0, num_tests)
  
  for(j in 1:num_tests) 
    pvalsdta[j]=pvalsdta[j]+sum(tmp[j]<A[,j])/nrow(A)    
  
  # Add p-values from direct p-value methods and chi-square methods.
  if(Continuous) {
    pvalsdta=c(pvalsdta, TS_cont_pval(x, y)$p.values) 
    
    if(length(nbins)==1) nbins=c(nbins, nbins)
    
    if(Dim==2) 
      chitmp=chisq2D_test_cont(x, y, Ranges, nbins, minexpcount)$p.values
    else 
      chitmp=rep(0, 2)
    
    pvalsdta=c(pvalsdta, chitmp)
    names(pvalsdta)=all.methods$cont
  }   
  else {
    pvalsdta=c(pvalsdta, chisq2D_test_disc(dta, minexpcount)$p.values)
    names(pvalsdta)=all.methods$disc
  }
  
  # Simulate p-values under the null to estimate the distribution
  # of the minimum p-value across selected tests.
  if(maxProcessor==1) {
    tmp=simpvals(dta, TS, typeTS, TSextra, A, Continuous, 
                 Ranges, nbins, minexpcount, B[2])
    
    pvalsTS=tmp$pvalsTS
    pvalsOther=tmp$pvalsOther
    pvalsChi=tmp$pvalsChi
  }
  else {
    cl2=parallel::makeCluster(maxProcessor)
    on.exit(parallel::stopCluster(cl2), add=TRUE)
    
    z=parallel::clusterCall(cl2, simpvals, dta, TS, 
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
    
    # Update B to actual number of p-value simulations.
    B[1]=nrow(pvalsTS)
  }
  
  # Choose default methods if the user did not specify doMethods.
  if(missing(doMethods)) {
    if(Continuous) doMethods=default.methods[["cont"]]
    else doMethods=default.methods[["disc"]]
  }
  
  # Expand "all" to the full method list.
  if(doMethods[1]=="all"){
    if(Continuous) doMethods=all.methods[["cont"]]
    else doMethods=all.methods[["disc"]]
  }
  
  # ES and EP are only available for 2D continuous data.
  if(Continuous & Dim>2) {
    doMethods=doMethods[doMethods!="ES"]
    doMethods=doMethods[doMethods!="EP"]
  }
  
  # Combine simulated p-values from all method classes.
  pvals=cbind(pvalsTS, pvalsOther, pvalsChi) 
  
  # Keep only selected methods.
  pvals=pvals[ ,doMethods,drop=FALSE]
  pvalsdta=pvalsdta[doMethods]
  
  # Observed minimum p-value across selected tests.
  minp_x=min(pvalsdta)
  
  # Simulated minimum p-values under the null.
  minp_sim=apply(pvals[, ,drop=FALSE], 1, min)
  
  # Estimate adjusted p-value from empirical CDF of min p-values.
  z=seq(0, 1, length=250)
  y=z
  
  for(i in 1:250) 
    y[i]=sum(minp_sim<=z[i])/length(minp_sim)
  
  # Linear interpolation at the observed minimum p-value.
  I=c(1:250)[z>minp_x][1]-1
  slope=(y[I+1]-y[I])/(z[I+1]-z[I])
  minp_adj=round(y[I]+slope*(minp_x-z[I]),4)
  
  # Print individual and adjusted p-values.
  message("p values of individual tests:")
  
  for(i in seq_along(pvalsdta)) 
    message(paste(names(pvalsdta)[i],": ", round(pvalsdta[i],4)))
  
  message(paste0("adjusted p value of combined tests: ", minp_adj))
}
