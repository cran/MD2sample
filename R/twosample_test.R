#' Multivariate Two-Sample Tests
#' 
#' This function runs a number of two sample tests using Rcpp and parallel computing.
#' 
#' For details consult vignette("MD2sample","MD2sample")
#' 
#' @param  x  Continuous data: either a matrix of numbers, or a list with two matrices called x and y.
#'                             if it is a matrix Observations are in different rows.
#'            Discrete data: a vector of counts or a matrix with columns named vals_x, vals_y, x and y.
#' @param  y a matrix of numbers if data is continuous or a vector of counts  if data is discrete. 
#' @param  vals_x =NA, a vector of values for discrete random variables, or NA if data is continuous.
#' @param  vals_y =NA, a vector of values for discrete random variables, or NA if data is continuous.
#' @param  TS user supplied routine to calculate test statistics for new tests.
#' @param  TSextra (optional) additional info passed to TS, if necessary.
#' @param  B =5000, number of simulation runs for permutation test.
#' @param  nbins =c(5,5), for chi square tests (2D only).
#' @param  minexpcount =5, lowest required count for chi-square test (2D only).
#' @param  Ranges =matrix(c(-Inf, Inf, -Inf, Inf),2,2), a 2x2 matrix with lower and upper bounds (2D only).
#' @param  DoTransform =TRUE, should data be transformed to unit hypercube?
#' @param  samplingmethod ="Binomial" for Binomial sampling or "independence" for independence sampling.
#' @param  rnull function to generate new data sets for simulation as an alternative to the permutation method.
#' @param  SuppressMessages =FALSE, should informative messages be printed?
#' @param  LargeSampleOnly =FALSE, should only methods with large sample theories be run?
#' @param  maxProcessor  number of cores to use. If missing the number of physical cores-1 
#'             is used. If set to 1 no parallel processing is done.
#' @param  doMethods ="all", Which methods should be included?
#' @return A list of two numeric vectors, the test statistics and the p values. 
#' @examples
#' #Two continuous data sets from a multivariate normal:
#' x = mvtnorm::rmvnorm(100, c(0,0))
#' y = mvtnorm::rmvnorm(120, c(0,0))
#' twosample_test(x, y, B=100, maxProcessor=1)
#' #Using a new test, this one is an (included) chi square test. 
#' #Also enter data as a list:
#' TSextra=list(which="statistics", nbins=rbind(c(3,3), c(4,4)))
#' dta=list(x=x, y=y)
#' twosample_test(dta, TS=chiTS.cont, TSextra=TSextra, B=100, maxProcessor=1)
#' #Two discrete data sets from some distribution:
#' x = table(sample(1:4, size=1000, replace = TRUE))
#' y = table(sample(1:4, size=1000, replace = TRUE, prob=c(1,2,1,1)))
#' vals_x=rep(1:2,2)
#' vals_y=rep(1:2, each=2)
#' twosample_test(x, y, vals_x, vals_y, B=100, maxProcessor=1)
#' #Run a discrete chi square test and enter the data as a matrix:
#' TSextra=list(which="statistics")
#' dta=cbind(x=x, y=y, vals_x=vals_x, vals_y=vals_y)
#' twosample_test(dta, TS=chiTS.disc, TSextra=TSextra, B=100, maxProcessor=1)
#' @export 
twosample_test = function(x, y, vals_x=NA, vals_y=NA, TS, TSextra, B=5000, 
                          nbins=c(5,5), minexpcount=5, 
                          Ranges=matrix(c(-Inf, Inf, -Inf, Inf),2,2),
                          DoTransform=TRUE, samplingmethod="Binomial", rnull,
                          SuppressMessages=FALSE, LargeSampleOnly=FALSE, 
                          maxProcessor, doMethods="all") {
  
  # Convert sampling method from character label to internal numeric code.
  # independence = 1, Binomial = 2.
  if(!is.numeric(samplingmethod))
    samplingmethod = ifelse(samplingmethod=="independence", 1, 2)

  # Determine whether data were supplied as separate x/y objects
  # or bundled inside x.
  if(missing(y)) {
    if(is.list(x)) { 
      # Continuous data supplied as list(x=..., y=...).
      if(!SuppressMessages) message("Data is assumed to be continuous")
      Continuous = TRUE
      dta = x
      y = x$y
      x = x$x 
      Dim = ncol(x)
    }
    else {
      # Discrete data supplied as a matrix with columns:
      # x, y, vals_x, vals_y.
      if(!SuppressMessages) message("Data is assumed to be discrete")
      Continuous = FALSE
      dta = x
      
      # Dummy matrices are used so later continuous-data references
      # do not fail, although the actual data are stored in dta.
      x = matrix(1:4,2,2)
      y = matrix(1:4,2,2)
      
      dta = list(
        x = dta[,"x"], 
        y = dta[,"y"], 
        vals_x = dta[,"vals_x"], 
        vals_y = dta[,"vals_y"]
      )
      
      # Discrete data are not transformed to the unit hypercube.
      DoTransform = FALSE
    }
  }
  else {
    
    # If vals_x or vals_y is NA, treat the data as continuous.
    # Otherwise treat them as discrete.
    Continuous = ifelse(any(is.na(c(vals_x, vals_y))), TRUE, FALSE)
    
    if(Continuous) {
      if(!SuppressMessages) message("Data is assumed to be continuous")
      dta = list(x=x, y=y)
      Dim = ncol(x)
    } 
    else {
      if(!SuppressMessages) message("Data is assumed to be discrete")
      dta = list(x=x, y=y, vals_x=vals_x, vals_y=vals_y)
      
      # Dummy matrices for compatibility with later references.
      x = matrix(1:4,2,2)
      y = matrix(1:4,2,2)
      
      DoTransform = FALSE
    }    
  }
  
  # Check whether the requested methods are valid for the data type.
  # If test_methods returns TRUE, stop and return NULL.
  if(test_methods(doMethods, Continuous)) return(NULL)
  
  if(Continuous) {
    
    # Ensure x is the smaller sample and y is the larger sample.
    if(nrow(y) < nrow(x)) {
      tmp = y
      y = x
      x = tmp
      dta = list(x=x, y=y)
    }
    
    # Store original data before optional transformation.
    rawdta = dta
    
    # Transform continuous observations to the unit hypercube.
    if(DoTransform) {
      dta = transform01(dta)
      x = dta$x
      y = dta$y
      Ranges = matrix(c(0, 1, 0, 1),2,2)
    }
  }
  
  # If no extra arguments are supplied for the test statistic,
  # create a placeholder list.
  if(missing(TSextra)) TSextra = list(aaa=0)
  
  # Add helper functions and pre-computed quantities needed by
  # continuous-data test statistics.
  if(Continuous) 
    TSextra = c(
      TSextra, 
      knn = function(x) FNN::get.knn(x, 5)$nn.index,
      dist = function(dta) find_dist(dta),
      distances = list(find_dist(dta)),
      DoTransform = DoTransform
    )
  else 
    # Add helper functions and settings needed by discrete-data tests.
    TSextra = c(
      TSextra, 
      dist = function(dta) NULL,
      organize = function(dta) dta = dta[order(dta[,1], dta[,2]), ],
      samplingmethod = samplingmethod
    )
  
  # If a null-data generator is provided, pass it to the test routine.
  # For continuous data, also pass the un-transformed original data.
  if(!missing(rnull)) {
    if(Continuous) 
      TSextra = c(TSextra, rnull=rnull, rawdta=list(rawdta))
    else 
      TSextra = c(TSextra, rnull=rnull)
  }   
  
  # Initialize objects that may later store chi-square or analytic p-value results.
  outchi = list(statistics=NULL, p.value=NULL)
  outpvals = list(statistics=NULL, p.value=NULL)
  
  # Decide whether to run built-in methods or a user-supplied statistic.
  CustomTS = TRUE
  
  if(missing(TS)) {
    
    # No user-supplied test statistic: use built-in methods.
    CustomTS = FALSE
    
    if(Continuous) {
      
      # For two-dimensional continuous data, run chi-square tests
      # if no custom null generator was supplied.
      if(Dim == 2) {
        if(length(nbins) == 1) nbins = c(nbins, nbins)
        
        if(missing(rnull))
          outchi = chisq2D_test_cont(x, y, Ranges, nbins, minexpcount)
      }  
      
      # Run built-in methods with large-sample p-values.
      if(missing(rnull))
        outpvals = TS_cont_pval(x, y)
      
      # typeTS identifies the calling convention used by calcTS/testC.
      typeTS = 1
      TS = TS_cont
      dta = list(x=x, y=y)
    }
    else {
      
      # Built-in discrete-data tests.
      typeTS = 4
      
      if(missing(rnull))
        outchi = chisq2D_test_disc(dta, minexpcount)
      
      TS = TS_disc
    }    
  }  
  else {
    
    # User supplied a custom test statistic.
    
    # If TS is an Rcpp/.Call routine, parallel execution is disabled.
    if(substr(deparse(TS)[2], 1, 5) == ".Call") {
      if(!missing(maxProcessor) & maxProcessor > 1) {
        if(!SuppressMessages) 
          message("Parallel Programming is not possible if custom TS is written in C++. Switching to single processor")  
        maxProcessor = 1
      }  
    }
    
    # Determine the expected calling convention from the number
    # of formal arguments in the supplied test statistic.
    if(Continuous) 
      typeTS = length(formals(TS))
    else 
      typeTS = ifelse(length(formals(TS)) == 5, 6, 5)  
  }
  
  # Compute the observed test statistic.
  TS_data = calcTS(dta, TS, typeTS, TSextra)
  
  # If B=0, return only the observed statistic without simulation p-values.
  if(B == 0) return(TS_data)
  
  # Require the test statistic output to be a named vector.
  if(any(is.null(names(TS_data)))) {
    if(!SuppressMessages) message("output of TS routine has to be a named vector!")
    return(NULL)
  }
  
  # Choose number of processors for simulation.
  # If not specified, use physical cores minus one, with minimum 1.
  if(missing(maxProcessor))
    maxProcessor = max(parallel::detectCores(logical = FALSE)-1, 1)  
  
  # Check whether parallelization is worthwhile.
  if(maxProcessor > 1) {
    tm = timecheck(dta, TS, typeTS, TSextra)
    
    # Use one processor if the task is too small for parallel overhead
    # to be worthwhile.
    if(2*tm[1]*B < 20 || B < 2*maxProcessor) {
      maxProcessor = 1
      if(!SuppressMessages) message("maxProcessor set to 1 for faster computation")
    }
    else if(!SuppressMessages) 
      message(paste("Using ", maxProcessor, " cores.."))  
  } 
  
  # Run simulation/permutation tests unless only large-sample methods
  # were requested.
  if(!LargeSampleOnly) {
    
    if(maxProcessor == 1) {
      # Serial simulation.
      outTS = testC(dta, TS, typeTS, TSextra, B=B)
    }
    else {
      # Parallel simulation. Each worker runs approximately B/maxProcessor
      # simulations, and the resulting p-values are averaged.
      cl = parallel::makeCluster(maxProcessor)
      on.exit(parallel::stopCluster(cl), add=TRUE)
      z = parallel::clusterCall(
        cl, testC, 
        dta=dta, TS=TS, typeTS=typeTS, TSextra=TSextra, 
        B=round(B/maxProcessor)
      )
      
      # Average p-values across workers.
      p = z[[1]]$p.values
      for(i in 2:maxProcessor) p = p + z[[i]]$p.values
      p = round(p/maxProcessor, 4)  
      
      outTS = list(statistics=z[[1]]$statistics, p.values=p)
    }
  }
  else {
    # Skip simulation-based tests.
    outTS = list(statistics=NULL, p.values=NULL)
  }
  
  # If the user supplied a custom test statistic, return only those results.
  if(CustomTS) return(signif.digits(outTS))
  
  # Combine simulation-based, analytic, and chi-square results.
  s = c(outTS$statistics, outpvals$statistics, outchi$statistic)
  p = c(outTS$p.values, outpvals$p.values, outchi$p.value)
  
  # If requested, keep only selected methods.
  if(doMethods[1] != "all") {
    s = s[doMethods]
    p = p[doMethods]
  }  
  
  # Return rounded/significant-digit formatted results.
  signif.digits(list(statistics=s, p.values=p))
}


