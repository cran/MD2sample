#' Helper function to find test statistics of simulated data.
#' @param  dta a list
#' @param TS test statistic routine
#' @param typeTS type of routine
#' @param TSextra a list
#' @param B number of simulation runs
#' @return a matrix
#' @keywords internal
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
      colnames(pvalsChi)=c("ChiSquare")
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

#' Adjusted p values for multivariate two-sample tests
#'
#' Runs several two-sample tests and returns their individual p-values together
#' with a simulation-based p-value for the minimum p-value across the selected tests.
#'
#' @param x,y,vals_x,vals_y Data arguments as in \code{twosample_test()}.
#' @param B Length-one or length-two vector giving simulation sizes.
#' @param nbins,minexpcount,Ranges Chi-square settings.
#' @param samplingmethod Sampling method for discrete data.
#' @param DoTransform Should continuous data be transformed to the unit hypercube?
#' @param rnull Optional parametric-bootstrap generator.
#' @param SuppressMessages Suppress informative messages?
#' @param maxProcessor Number of processors.
#' @param doMethods Methods to combine. If missing, a default subset is used.
#' @param seed Optional random-number seed for reproducibility.
#' @return A named numeric vector containing the individual p-values and the
#'   adjusted minimum-p value in the final element, named \code{"Min p"}.
#' @export
twosample_test_adjusted_pvalue <- function(x, y, vals_x=NA, vals_y=NA,
                                            B=c(5000,1000), nbins=c(5,5),
                                            minexpcount=5, samplingmethod="Binomial",
                                            Ranges=matrix(c(-Inf,Inf,-Inf,Inf),2,2),
                                            DoTransform=TRUE, rnull,
                                            SuppressMessages=FALSE, maxProcessor,
                                            doMethods, seed=NULL) {
  if(length(B)==1L) B <- c(B,B)
  if(length(B)!=2L) stop("B must have length 1 or 2.", call.=FALSE)
  for(i in seq_along(B)) validate_integer_scalar(B[i], paste0("B[",i,"]"), 1L)
  if(!missing(maxProcessor)) validate_integer_scalar(maxProcessor, "maxProcessor", 1L)
  validate_logical_scalar(DoTransform, "DoTransform")
  validate_logical_scalar(SuppressMessages, "SuppressMessages")
  samplingmethod <- normalize_samplingmethod(samplingmethod)
  if(!is.null(seed)) {
    if(length(seed)!=1L || is.na(seed) || !is.finite(seed))
      stop("seed must be NULL or a single finite number", call.=FALSE)
    set.seed(seed)
  }

  inp <- prepare_twosample_input(
    x=x, y_missing=missing(y), y=if(missing(y)) NULL else y,
    vals_x=vals_x, vals_y=vals_y, DoTransform=DoTransform,
    Ranges=Ranges, SuppressMessages=SuppressMessages
  )
  Continuous <- inp$Continuous; dta <- inp$dta
  x <- inp$x; y <- inp$y; Dim <- inp$Dim; Ranges <- inp$Ranges
  if(length(nbins)==1L) nbins <- rep(nbins,2L)

  rnull0 <- if(missing(rnull)) NULL else rnull
  TSextra <- makeTSextra(dta, Continuous, inp$DoTransform, samplingmethod,
                         NULL, rnull0, inp$rawdta)
  # Built-in calling conventions are type 1 (continuous) and type 4 (discrete).
  typeTS <- if(Continuous) 1L else 4L
  TS <- if(Continuous) TS_cont else TS_disc
  TS_data <- calcTS(dta, TS, typeTS, TSextra)
  validate_ts_output(TS_data)

  outOther <- if(Continuous) TS_cont_pval(x,y) else list(statistics=NULL,p.values=NULL)
  outChi <- if(Continuous && Dim==2L) chisq2D_test_cont(x,y,Ranges,nbins,minexpcount) else
            if(!Continuous) chisq2D_test_disc(dta,minexpcount) else list(statistics=NULL,p.values=NULL)
  observed_names <- c(names(TS_data), names(outOther$p.values), names(outChi$p.values))
  defaultMethods <- if(Continuous) c("ES","CvM","AZ","NN5","BG") else c("ChiSquare","KS","AZ","CvM")
  allMethods <- observed_names
  if(missing(doMethods)) doMethods <- intersect(defaultMethods, allMethods)
  if(length(doMethods)==1L && identical(doMethods,"all")) doMethods <- allMethods
  bad <- setdiff(doMethods, allMethods)
  if(length(bad)) stop("Unknown method(s): ", paste(bad, collapse=", "), call.=FALSE)

  maxProcessor <- makemaxProcessor(maxProcessor, dta, TS, typeTS, TSextra, B[1],
                                   SuppressMessages, FALSE)
  B[1] <- ceiling(B[1]/maxProcessor)*maxProcessor
  B[2] <- ceiling(B[2]/maxProcessor)*maxProcessor

  if(maxProcessor==1L) {
    A <- simTS(dta, TS, typeTS, TSextra, B[1])
  } else {
    cl1 <- parallel::makeCluster(maxProcessor)
    on.exit(parallel::stopCluster(cl1), add=TRUE)
    if(!is.null(seed)) parallel::clusterSetRNGStream(cl1, iseed=seed)
    z <- parallel::clusterCall(cl1, simTS, dta, TS, typeTS, TSextra,
                               as.integer(B[1]/maxProcessor))
    A <- do.call(rbind,z)
  }

  pvalsTS <- vapply(seq_along(TS_data), function(j) mean(TS_data[j] < A[,j]), numeric(1))
  names(pvalsTS) <- names(TS_data)
  pvalsdta <- c(pvalsTS, outOther$p.values, outChi$p.values)

  if(maxProcessor==1L) {
    tmp <- simpvals(dta, TS, typeTS, TSextra, A, Continuous,
                    Ranges, nbins, minexpcount, B[2])
    pvalsTSsim <- tmp$pvalsTS; pvalsOther <- tmp$pvalsOther; pvalsChi <- tmp$pvalsChi
  } else {
    cl2 <- parallel::makeCluster(maxProcessor)
    on.exit(parallel::stopCluster(cl2), add=TRUE)
    if(!is.null(seed)) parallel::clusterSetRNGStream(cl2, iseed=seed+1)
    z <- parallel::clusterCall(cl2, simpvals, dta, TS, typeTS, TSextra, A,
                               Continuous, Ranges, nbins, minexpcount,
                               as.integer(B[2]/maxProcessor))
    pvalsTSsim <- do.call(rbind,lapply(z,`[[`,1))
    pvalsOther <- if(Continuous) do.call(rbind,lapply(z,`[[`,2)) else NULL
    pvalsChi <- do.call(rbind,lapply(z,`[[`,3))
  }
  pvals <- cbind(pvalsTSsim, pvalsOther, pvalsChi)
  pvals <- pvals[,doMethods,drop=FALSE]
  pvalsdta <- pvalsdta[doMethods]
  minp_x <- min(pvalsdta, na.rm=TRUE)
  minp_sim <- apply(pvals,1,min,na.rm=TRUE)
  minp_adj <- round(mean(minp_sim <= minp_x, na.rm=TRUE),4)

  if(!SuppressMessages) {
    message("p values of individual tests:")
    for(i in seq_along(pvalsdta)) message(names(pvalsdta)[i], ": ", round(pvalsdta[i],4))
    message("adjusted p value of combined tests: ", minp_adj)
  }
  c(pvalsdta, "Min p"=minp_adj)
}
