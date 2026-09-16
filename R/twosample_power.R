#' Power Estimation for Multivariate Two-Sample Tests
#' 
#' Estimate the power of various two sample tests using Rcpp and parallel computing.
#' 
#' For details consult vignette("MD2sample","MD2sample")
#' 
#' @param  f  function to generate a list with data sets x and y for continuous data or
#'         a four-column numeric matrix for discrete data. Standard names vals_x, vals_y, x and y are used when present; otherwise the two support/bin and two count columns are inferred from their numbers of distinct values.
#' @param  ... additional arguments passed to f, up to 2.
#' @param  TS routine to calculate test statistics for new tests. For discrete data it may have signature \code{TS(x)} or \code{TS(x, TSextra)}, where \code{x} is the four-column discrete-data matrix; the legacy 4- or 5-argument form is also accepted.
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
#' @param CI =FALSE if TRUE, return Monte Carlo standard errors and Wilson confidence intervals.
#' @param conf.level =0.95 confidence level used when CI=TRUE.
#' @param seed =NULL optional random-number seed for reproducibility.
#' @return By default, a numeric matrix or vector of power values. If CI=TRUE, an object of class \code{"MD2sample_power"} with power estimates, Monte Carlo standard errors, and Wilson confidence limits.
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
                         LargeSampleOnly=FALSE, maxProcessor, doMethods ="all",
                         CI=FALSE, conf.level=0.95, seed=NULL) {
  
  if(!is.function(f)) stop("f must be a function.", call.=FALSE)
  validate_probability(alpha, "alpha")
  validate_integer_scalar(B, "B", 1L)
  validate_logical_scalar(With.p.value, "With.p.value")
  validate_logical_scalar(DoTransform, "DoTransform")
  validate_logical_scalar(SuppressMessages, "SuppressMessages")
  validate_logical_scalar(LargeSampleOnly, "LargeSampleOnly")
  validate_logical_scalar(CI, "CI")
  if(!is.numeric(conf.level) || length(conf.level)!=1L || is.na(conf.level) || conf.level<=0 || conf.level>=1)
    stop("conf.level must be a single number between 0 and 1", call.=FALSE)
  if(!missing(maxProcessor)) validate_integer_scalar(maxProcessor, "maxProcessor", 1L)
  if(!is.null(seed)) {
    if(length(seed)!=1L || is.na(seed) || !is.finite(seed))
      stop("seed must be NULL or a single finite number", call.=FALSE)
    set.seed(seed)
  }
  samplingmethod <- normalize_samplingmethod(samplingmethod)
  
  # Create a wrapper rxy(a,b) around the user-supplied generator f().
  # This standardizes f so later code can always call it with two arguments.
  ldots <- list(...)
  nldots <- length(ldots)
  if(nldots > 2L) stop("At most two parameter vectors may be supplied through ...", call.=FALSE)
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
      stop("lengths of parameter vectors are not compatible.", call.=FALSE)
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
  
  if(is.matrix(dta) || is.data.frame(dta)) {
    # A matrix returned by f() represents discrete data.  Standardize the
    # inferred column roles once in rxy so every later simulation, including
    # the compiled power routine, receives vals_x, vals_y, x, y in that order.
    if(!is_discrete_data(dta))
      stop("A matrix returned by f must have a recognizable four-column discrete-data format.", call.=FALSE)
    rxy0 <- rxy
    rxy <- function(a=0, b=0) prepare_discrete_data(rxy0(a,b), as_list=FALSE)
    dta <- prepare_discrete_data(dta)
    
    # Dummy matrices used only for compatibility with later references.
    x=matrix(1:4,2,2)
    y=matrix(1:4,2,2)
    
    Continuous=FALSE
    DoTransform=FALSE
  }
  
  # Verify that requested methods are valid for this data type.
  test_methods(doMethods, Continuous)
  
  if(Continuous) {
    x=dta$x
    y=dta$y
    Dim=ncol(x)
    
    # This routine requires nrow(x) <= nrow(y).
    if(nrow(y)<nrow(x)) { 
      stop("For continuous power studies, the generated x sample must not be larger than y.", call.=FALSE)
    }
  }  
  
  # Optionally transform continuous data to the unit hypercube.
  if(DoTransform) {
    dta=transform01(dta)
    x=dta$x
    y=dta$y
    Ranges=matrix(c(0, 1, 0, 1),2,2)
  }
  
  # Build auxiliary information for test-statistic routines.
  TSextra0 <- if(missing(TSextra)) NULL else TSextra
  rnull0 <- if(missing(rnull)) NULL else rnull
  TSextra <- makeTSextra(dta, Continuous, DoTransform, samplingmethod,
                         TSextra0, rnull0, dta)
  
  # Placeholder for chi-square or other large-sample power results.
  pwrchi=NULL
  
  # Select built-in or user-supplied test statistic.
  tmpTS <- maketypeTS(TS, Continuous)
  typeTS <- tmpTS$typeTS
  TS <- tmpTS$TS
  CustomTS <- tmpTS$CustomTS
  if(With.p.value && !CustomTS)
    stop("With.p.value=TRUE requires a user-supplied TS.", call.=FALSE)
  if(LargeSampleOnly && CustomTS)
    stop("LargeSampleOnly=TRUE is only available for the included methods.", call.=FALSE)
  TS_data <- calcTS(dta, TS, typeTS, TSextra)
  validate_ts_output(TS_data)
  
  # Decide whether parallel computation is worthwhile.
  # Direct p-value power and C++ custom statistics use one processor.
  if(With.p.value || tmpTS$useSingleProcessor) maxProcessor=1
  
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
    labels <- if(nldots==0) rep("", nrow(pwr)) else if(nldots==1) as.character(avals) else paste0(avals,"|",bvals)
    return(.format_MD2sample_power(pwr, B, CI, conf.level, alpha, labels, seed))
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
      cl1=parallel::makeCluster(maxProcessor1)
      on.exit(parallel::stopCluster(cl1), add=TRUE)
      if(!is.null(seed)) parallel::clusterSetRNGStream(cl1, iseed=seed)
      z=parallel::clusterCall(cl1, powerC, 
                              rxy,  avals, bvals,
                              TS, typeTS, TSextra, round(B[1]/maxProcessor1))
      
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
      cl2 <- parallel::makeCluster(maxProcessor2)
      on.exit(parallel::stopCluster(cl2), add=TRUE)
      if(!is.null(seed)) parallel::clusterSetRNGStream(cl2, iseed=seed+1)
      u = parallel::clusterCall(cl2, power_pvals, 
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
  
  labels <- if(nldots==0) rep("", nrow(pwr)) else if(nldots==1) as.character(avals) else paste0(avals,"|",bvals)
  .format_MD2sample_power(pwr, B, CI, conf.level, alpha, labels, seed)
}

.format_MD2sample_power <- function(pwr, B, CI, conf.level, alpha, param_alt, seed) {
  if(!CI) {
    if(is.matrix(pwr) && nrow(pwr)==1L) pwr <- pwr[1, ]
    return(round(pwr, 4))
  }
  z <- stats::qnorm(1-(1-conf.level)/2)
  den <- 1+z^2/B
  center <- (pwr+z^2/(2*B))/den
  half <- z*sqrt(pwr*(1-pwr)/B + z^2/(4*B^2))/den
  lower <- pmax(center-half, 0)
  upper <- pmin(center+half, 1)
  mc.se <- sqrt(pwr*(1-pwr)/B)
  if(is.matrix(pwr) && nrow(pwr)==1L) {
    nm <- colnames(pwr)
    pwr <- as.numeric(pwr[1,]); lower <- as.numeric(lower[1,]); upper <- as.numeric(upper[1,]); mc.se <- as.numeric(mc.se[1,])
    names(pwr)<-nm; names(lower)<-nm; names(upper)<-nm; names(mc.se)<-nm
  }
  out <- list(power=round(pwr,4), mc.se=round(mc.se,4), lower=round(lower,4), upper=round(upper,4),
              B=B, conf.level=conf.level, interval="Wilson", alpha=alpha, param_alt=param_alt, seed=seed)
  class(out) <- "MD2sample_power"
  out
}

#' Methods for MD2sample power results
#'
#' Print and convert power estimates with Monte Carlo uncertainty.
#'
#' @param x An object of class \code{"MD2sample_power"}.
#' @param ... Further arguments passed to the underlying print or data-frame method.
#' @param row.names Ignored; included for compatibility with \code{as.data.frame}.
#' @param optional Ignored; included for compatibility with \code{as.data.frame}.
#' @return The print method returns \code{x} invisibly. The data-frame method
#'   returns the alternative parameter, method, power estimate, Monte Carlo
#'   standard error, and Wilson confidence limits.
#' @name MD2sample_power-methods
#' @export
print.MD2sample_power <- function(x, ...) {
  cat(sprintf("Estimated power with %.1f%% Wilson confidence intervals (B = %d)\n", 100*x$conf.level, x$B))
  if(is.matrix(x$power)) {
    for(i in seq_len(nrow(x$power))) {
      lab <- rownames(x$power)[i]
      if(is.null(lab) || !nzchar(lab)) lab <- x$param_alt[i]
      if(!is.null(lab) && nzchar(lab)) cat("\nAlternative parameter:", lab, "\n")
      out <- data.frame(power=x$power[i,], mc.se=x$mc.se[i,], lower=x$lower[i,], upper=x$upper[i,], row.names=colnames(x$power), check.names=FALSE)
      print(out, ...)
    }
  } else {
    print(data.frame(power=x$power, mc.se=x$mc.se, lower=x$lower, upper=x$upper, row.names=names(x$power), check.names=FALSE), ...)
  }
  invisible(x)
}

#' @rdname MD2sample_power-methods
#' @export
as.data.frame.MD2sample_power <- function(x, row.names=NULL, optional=FALSE, ...) {
  if(is.matrix(x$power)) {
    nr <- nrow(x$power); nc <- ncol(x$power)
    labs <- rownames(x$power); if(is.null(labs)) labs <- x$param_alt
    return(data.frame(param_alt=rep(labs, times=nc), method=rep(colnames(x$power), each=nr),
                      power=c(x$power), mc.se=c(x$mc.se), lower=c(x$lower), upper=c(x$upper),
                      row.names=NULL, check.names=FALSE))
  }
  data.frame(param_alt=rep(x$param_alt[1], length(x$power)), method=names(x$power), power=unname(x$power),
             mc.se=unname(x$mc.se), lower=unname(x$lower), upper=unname(x$upper), row.names=NULL, check.names=FALSE)
}
