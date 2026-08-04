#' Power Estimation for Multivariate Two-Sample Tests
#' 
#' Estimate the power of various two sample tests using Rcpp and parallel computing.
#' 
#' For details consult vignette("MD2sample","MD2sample")
#' 
#' @param  f  function to generate a list with data sets x and y for continuous data or
#'         a matrix with columns vals_x, vals_y, x and y for discrete data.
#' @param  ... additional arguments passed to f, up to 2.
#' @param  TS routine to calculate test statistics for new tests.
#' @param  TSextra additional info passed to TS, if necessary.
#' @param  alpha =0.05, the type I error probability of the hypothesis test. 
#' @param  B =1000, number of simulation runs.
#' @param  nbins =c(5, 5), number of bins for chi square test if Dim=2.
#' @param  minexpcount =5, lowest required count for chi-square test.
#' @param  Ranges =matrix(c(-Inf, Inf, -Inf, Inf),2,2), a 2x2 matrix with lower and upper bounds.
#' @param  samplingmethod ="Binomial" for Binomial sampling or "independence" for independence 
#'         sampling in the discrete data case.
#' @param  rnull function to generate new data sets for parametric bootstrap.
#' @param  With.p.value =FALSE, does user supplied routine return p values?
#' @param  DoTransform =TRUE, should data be transformed to  to unit hypercube? 
#' @param  SuppressMessages =FALSE, should messages be printed?
#' @param  LargeSampleOnly =FALSE, should only methods with large sample theories be run?
#' @param  maxProcessor number of cores to use. If missing the number of physical cores-1 
#'             is used. If set to 1 no parallel processing is done.
#' @param  doMethods ="all", which methods should be included?
#' @return A numeric matrix or vector of power values.
#' @examples
#' #Note that the resulting power estimates are meaningless because
#' #of the extremely low number of simulation runs B, required because of CRAN timing rule
#' #
#' #Power of tests when one data set comes from a standard normal multivariate distribution function
#' #and the other data set from a multivariate normal with correlation
#' #number of simulation runs is ridiculously small because of CRAN submission rules
#' f=function(a=0) {
#'  S=diag(2) 
#'  x=mvtnorm::rmvnorm(100, sigma = S)
#'  S[1,2]=a
#'  S[2,1]=a
#'  y=mvtnorm::rmvnorm(120, sigma = S)
#'  list(x=x, y=y)
#' }
#' twosample_power(f, c(0, 0.5), B=10, maxProcessor=1)
#' #Power of use supplied test. Example is a (included) chi-square test:
#' TSextra=list(which="statistics", nbins=rbind(c(3,3), c(4,4)))
#' twosample_power(f, c(0, 0.5), TS=chiTS.cont, TSextra=TSextra, B=10, maxProcessor=1)
#' #Same example, but this time the user supplied routine calculates p values:
#' TSextra=list(which="pvalues", nbins=c(4,4))
#' twosample_power(f, c(0, 0.5), TS=chiTS.cont, TSextra=TSextra, B=10, 
#'              With.p.value=TRUE, maxProcessor=1)
#' #Example for discrete data
#' g=function(p1, p2) {
#'   x = table(sample(1:4, size=1000, replace = TRUE))
#'   y = table(sample(1:4, size=500, replace = TRUE, prob=c(p1,p2,1,1)))
#'   cbind(vals_x=rep(1:2,2),  vals_y=rep(1:2, each=2), x=x, y=y)
#' }  
#' twosample_power(g, 1.5, 1.6, B=10, maxProcessor=1)
#' @export 
twosample_power=function(f, ..., TS, TSextra, alpha=0.05, B=1000, 
                         nbins=c(5,5), minexpcount =5, Ranges=matrix(c(-Inf, Inf, -Inf, Inf),2,2),
                         samplingmethod="Binomial", rnull, With.p.value=FALSE,
                         DoTransform=TRUE, SuppressMessages=FALSE, 
                         LargeSampleOnly=FALSE, maxProcessor, doMethods ="all") {
  
  # Convert sampling method to internal numeric code.
  # independence = 1, Binomial = 2.
  if(!is.numeric(samplingmethod))  
    samplingmethod=ifelse(samplingmethod=="independence", 1, 2)
  
  # Create a wrapper rxy(a,b) around the user-supplied generator f().
  # This standardizes f so later code can always call it with two arguments.
  ldots <- list(...)
  nldots <- length(ldots)
  if(nldots==0) {
    rxy=function(a=0, b=0) f()
    avals=0
    bvals=0
  }
  if(nldots==1) {
    rxy=function(a=0, b=0) f(a)
    avals=ldots[[1]]
    bvals=0
  }
  if(nldots==2) {
    rxy=function(a=0, b=0) f(a,b)
    avals=ldots[[1]]
    bvals=ldots[[2]]
  }
  
  # Check parameter-vector lengths.
  # If one vector is scalar, recycle it to match the other.
  # If both have length > 1 but different lengths, stop.
  if(length(avals)!=length(bvals)) {
    if(min(c(length(avals),length(bvals)))>1) {
      if(!SuppressMessages) message("lengths of parameter vectors not compatible!\n")
      return(NULL)
    }
    if(length(avals)==1) {
      avals=rep(avals, length(bvals))
    }    
    else bvals=rep(bvals, length(avals))
  }    
  
  # Generate one example data set to determine the data type
  # and initialize dimensions/settings.
  dta = rxy(avals[1], bvals[1])
  Continuous=TRUE
  
  if(is.matrix(dta)) {
    # Matrix output is interpreted as discrete data.
    # Expected columns: vals_x, vals_y, x counts, y counts.
    dta=list(x=dta[,3], y=dta[,4], vals_x=dta[,1], vals_y=dta[,2])
    
    # Dummy matrices used only for compatibility with later references.
    x=matrix(1:4,2,2)
    y=matrix(1:4,2,2)
    
    Continuous=FALSE
    DoTransform=FALSE
  }
  
  # Verify that requested methods are valid for this data type.
  if(test_methods(doMethods, Continuous)) return(NULL)
  
  if(Continuous) {
    x=dta$x
    y=dta$y
    Dim=ncol(x)
    
    # This routine requires nrow(x) <= nrow(y).
    if(nrow(y)<nrow(x)) { 
      if(!SuppressMessages) message("sample size of x should not be larger than sample size of y")
      return(NULL)
    }
  }  
  
  # Optionally transform continuous data to the unit hypercube.
  if(DoTransform) {
    dta=transform01(dta)
    x=dta$x
    y=dta$y
    Ranges=matrix(c(0, 1, 0, 1),2,2)
  }
  
  # Initialize optional extra arguments for test statistic routines.
  if(missing(TSextra)) TSextra=list(aaa=0)
  
  # Add helper functions and precomputed quantities for continuous data.
  if(Continuous)
    TSextra = c(TSextra, 
                knn=function(x) FNN::get.knn(x, 5)$nn.index,
                dist=function(dta) find_dist(dta),
                distances=list(find_dist(list(x=x,y=y))),
                DoTransform=DoTransform,
                ParametricBootstrap=FALSE)
  else 
    # Add helper functions and settings for discrete data.
    TSextra = c(TSextra, 
                dist=function(dta) NULL,
                organize=function(dta) dta=dta[order(dta[,1], dta[,2]), ],
                samplingmethod=samplingmethod,
                ParametricBootstrap=FALSE)
  
  # If a null generator is provided, use parametric bootstrap.
  if(!missing(rnull)) {
    TSextra$ParametricBootstrap=TRUE
    TSextra=c(TSextra, rnull=rnull, rawdta=list(dta))
  }   
  
  # Placeholder for chi-square or other large-sample power results.
  pwrchi=NULL
  
  # Select built-in or user-supplied test statistic.
  if(missing(TS)) {
    CustomTS=FALSE
    
    if(Continuous) {
      typeTS=1
      TS=TS_cont
    }
    else {
      typeTS=4
      TS=TS_disc
    }  
  }
  else {
    CustomTS=TRUE
    
    # Determine calling convention from number of formal arguments.
    if(Continuous) typeTS=length(formals(TS))
    else typeTS=ifelse(length(formals(TS))==5, 6, 5)
  }
  
  # Compute one observed statistic to validate output and get method names.
  TS_data=calcTS(dta, TS, typeTS, TSextra)
  
  if(is.null(names(TS_data))) {
    if(!SuppressMessages) message("output of TS routine has to be a named vector!")
    return(NULL)
  }  
  
  methodnames=names(TS_data)
  
  # Decide whether parallel computation is worthwhile.
  # With.p.value uses a different routine and is forced to one processor.
  if(With.p.value) maxProcessor=1
  
  if(missing(maxProcessor)) {
    ncores=max(parallel::detectCores(logical=FALSE)-1,1)
    tm=timecheck(dta, TS, typeTS, TSextra)
    
    # Ensure timing vector has two components.
    if(length(tm)==1) tm=c(tm,0)
    
    # Estimate total runtime for simulation-based and p-value-based parts.
    totaltime=2*tm*length(avals)*B
    
    if(max(totaltime)<20 | B<=2*ncores) 
      if(!SuppressMessages) message("maxProcessor set to 1 for faster computation")
    else if(!SuppressMessages) message(paste("Using ", ncores," cores..")) 
    
    maxProcessor1=1
    if(totaltime[1]>20 && B>2*ncores) 
      maxProcessor1=ncores
    
    maxProcessor2=1
    if(typeTS==1 && totaltime[2]>20 && B>2*ncores) 
      maxProcessor2=ncores
    
    # Print estimated total time when parallel computation is used.
    if(max(totaltime)>20 && B>2*ncores) {
      if(maxProcessor1==1 && maxProcessor2>1)
        totaltime=30+totaltime[2]/maxProcessor2
      if(maxProcessor1>1 && maxProcessor2==1)
        totaltime=30+totaltime[1]/maxProcessor1
      if(maxProcessor1>1 && maxProcessor2>1)
        totaltime=50+sum(totaltime)/maxProcessor1
      
      totaltime=round(totaltime,-1)
      timeunit="seconds"
      
      if(max(totaltime)>60) {
        totaltime=round(totaltime/60,1)
        timeunit="minutes"
      }
      
      if(!SuppressMessages) message(paste("estimated time:", totaltime , timeunit))
    }
  }  
  else {
    # User explicitly supplied number of processors.
    maxProcessor1=maxProcessor
    maxProcessor2=maxProcessor
  }
  
  # If requested, estimate power directly from returned p-values.
  if(With.p.value) {
    pwr=power_pvals(rxy, avals, bvals, TS=TS, typeTS, TSextra, alpha=alpha, B=B)
    
    if(nldots==0) rownames(pwr)=NULL
    if(nldots==1) rownames(pwr)=avals
    if(nldots==2) rownames(pwr)=paste0(avals,"|",bvals)  
    
    if(doMethods[1]!="all") pwr=pwr[, doMethods, drop=FALSE]
    
    return(round(pwr, 4))
  }
  
  # Estimate power using simulated null critical values.
  if(!LargeSampleOnly) {
    if(maxProcessor1==1) {
      tmp=powerC(rxy, avals, bvals, TS, typeTS, TSextra, B)
      Data=tmp$Data
      Simulated=tmp$Simulated
      paramalt=tmp$paramalt
    }    
    else {
      # Parallel simulation for empirical critical values.
      cl=parallel::makeCluster(maxProcessor1)
      on.exit(parallel::stopCluster(cl), add=TRUE)
      z=parallel::clusterCall(cl, powerC, 
                              rxy,  avals, bvals,
                              TS, typeTS, TSextra, round(B[1]/maxProcessor1))
      
      parallel::stopCluster(cl)
      
      # Combine results from all workers.
      Simulated=z[[1]][["Simulated"]]
      Data=z[[1]][["Data"]]
      paramalt=z[[1]][["paramalt"]]
      
      for(i in 2:maxProcessor1) {
        Simulated=rbind(Simulated, z[[i]][["Simulated"]])
        Data=rbind(Data, z[[i]][["Data"]])
        paramalt=rbind(paramalt,z[[i]][["paramalt"]])
      }  
    }
    
    # Compute empirical power for each parameter setting and method.
    pwr=matrix(0, length(avals), length(TS_data))
    colnames(pwr)=names(TS_data)
    
    for(i in seq_along(avals)) {
      Index=c(1:nrow(Data))[paramalt[,1]==avals[i]&paramalt[,2]==bvals[i]]
      
      tmpD=Data[Index, , drop=FALSE]
      tmpS=Simulated[Index, , drop=FALSE]
      
      # Critical value is the 1-alpha null quantile.
      crtval=apply(tmpS, 2, quantile, prob=1-alpha, na.rm=TRUE)
      
      # Power is the proportion of alternative statistics exceeding
      # the simulated critical value.
      for(j in seq_along(crtval)) 
        pwr[i, j]=sum(tmpD[ ,j]>crtval[j])/nrow(tmpD)
    }
  }  
  
  # Compute power for methods that return p-values directly,
  # such as large-sample or chi-square methods.
  pwrothers=NULL
  
  if(missing(rnull) & (typeTS %in% c(1, 4))) {
    if(maxProcessor2==1) {
      pwrothers=power_pvals(rxy, avals, bvals,  
                            TS=TS, typeTS=typeTS, TSextra=TSextra,
                            nbins=nbins, minexpcount=minexpcount, 
                            Ranges=Ranges, alpha=alpha, B=B)
    }  
    else { 
      # Parallel power calculation for p-value-based methods.
      cl <- parallel::makeCluster(maxProcessor2)
      on.exit(parallel::stopCluster(cl), add=TRUE)
      u = parallel::clusterCall(cl, power_pvals, 
                                rxy, avals, bvals,
                                TS=TS, typeTS=typeTS, TSextra=TSextra,
                                nbins=nbins, minexpcount=minexpcount, Ranges=Ranges,
                                alpha=alpha, B=round(B/maxProcessor2))
      
      # Average power estimates over workers.
      pwrothers=u[[1]]
      for(i in 2:maxProcessor2) pwrothers=pwrothers+u[[i]]
      pwrothers = pwrothers/maxProcessor2
    }
  }
  
  # Combine simulation-based and p-value-based power estimates.
  if(LargeSampleOnly) pwr=pwrothers
  else if(!CustomTS) pwr = cbind(pwr, pwrothers)
  
  # Assign row names based on the number of tuning parameters supplied.
  if(nldots==0) rownames(pwr)=NULL
  if(nldots==1) rownames(pwr)=avals
  if(nldots==2) rownames(pwr)=paste0(avals,"|",bvals)
  
  # Keep only requested methods.
  if(doMethods[1]!="all") pwr=pwr[, doMethods, drop=FALSE]
  
  # Return rounded power estimates.
  round(pwr, 4)
}
