#' Multivariate Two-Sample Tests
#' 
#' This function runs a number of two sample tests using Rcpp and parallel computing.
#' 
#' For details consult vignette("MD2sample","MD2sample")
#' 
#' @param  x  Continuous data: either a matrix of numbers, or a list with two matrices called x and y.
#'                             if it is a matrix Observations are in different rows.
#'            Discrete data: a vector of counts or a four-column numeric matrix containing two support/bin columns and two count columns. Standard names vals_x, vals_y, x and y are used when present; otherwise the roles are inferred from the numbers of distinct values.
#' @param  y a matrix of numbers if data is continuous or a vector of counts  if data is discrete. 
#' @param  vals_x =NA, a vector of values for discrete random variables, or NA if data is continuous.
#' @param  vals_y =NA, a vector of values for discrete random variables, or NA if data is continuous.
#' @param  TS user supplied routine to calculate test statistics for new tests. For discrete data it may have signature \code{TS(x)} or \code{TS(x, TSextra)}, where \code{x} is the four-column discrete-data matrix; the legacy 4- or 5-argument form is also accepted.
#' @param  TSextra (optional) additional info passed to TS, if necessary.
#' @param  B =5000, number of simulation runs for permutation test.
#' @param  seed =NULL, optional seed for simulations
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
#' @return An object of class \code{"MD2sample_test"} with components \code{results}, \code{statistics}, \code{p.values}, \code{metadata}, and \code{call}. The traditional \code{$statistics} and \code{$p.values} components are retained for backward compatibility.
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
twosample_test <- function(x, y, vals_x=NA, vals_y=NA, TS, TSextra, B=5000,
                           seed=NULL, nbins=c(5,5), minexpcount=5,
                           Ranges=matrix(c(-Inf, Inf, -Inf, Inf),2,2),
                           DoTransform=TRUE, samplingmethod="Binomial", rnull,
                           SuppressMessages=FALSE, LargeSampleOnly=FALSE,
                           maxProcessor, doMethods="all") {
  call <- match.call()
  B_requested <- B
  validate_integer_scalar(B, "B", 0L)
  validate_logical_scalar(DoTransform, "DoTransform")
  validate_logical_scalar(SuppressMessages, "SuppressMessages")
  validate_logical_scalar(LargeSampleOnly, "LargeSampleOnly")
  if(!missing(maxProcessor)) validate_integer_scalar(maxProcessor, "maxProcessor", 1L)
  samplingmethod <- normalize_samplingmethod(samplingmethod)
  if(!is.null(seed)) {
    if(length(seed) != 1L || is.na(seed) || !is.finite(seed))
      stop("seed must be NULL or a single finite number", call.=FALSE)
    set.seed(seed)
  }
  inp <- prepare_twosample_input(
    x=x, y_missing=missing(y), y=if(missing(y)) NULL else y,
    vals_x=vals_x, vals_y=vals_y, DoTransform=DoTransform,
    Ranges=Ranges, SuppressMessages=SuppressMessages
  )
  Continuous <- inp$Continuous
  dta <- inp$dta
  x <- inp$x; y <- inp$y; Dim <- inp$Dim
  DoTransform <- inp$DoTransform; Ranges <- inp$Ranges

  test_methods(doMethods, Continuous)
  if(length(nbins) == 1L) nbins <- rep(nbins, 2L)
  if(any(!is.finite(nbins)) || any(nbins < 1) || any(abs(nbins-round(nbins)) > sqrt(.Machine$double.eps)))
    stop("nbins must contain positive integers.", call.=FALSE)
  if(length(minexpcount) != 1L || is.na(minexpcount) || !is.finite(minexpcount) || minexpcount <= 0)
    stop("minexpcount must be a single positive number.", call.=FALSE)

  TSextra0 <- if(missing(TSextra)) NULL else TSextra
  rnull0 <- if(missing(rnull)) NULL else rnull
  TSextra <- makeTSextra(dta, Continuous, DoTransform, samplingmethod,
                         TSextra0, rnull0, inp$rawdta)

  outchi <- list(statistics=NULL, p.values=NULL)
  outpvals <- list(statistics=NULL, p.values=NULL)
  tmp <- maketypeTS(TS, Continuous)
  typeTS <- tmp$typeTS; TS <- tmp$TS; CustomTS <- tmp$CustomTS

  if(CustomTS && LargeSampleOnly)
    stop("LargeSampleOnly=TRUE is only available for the included methods.", call.=FALSE)

  if(!CustomTS) {
    if(Continuous) {
      if(Dim == 2L && missing(rnull))
        outchi <- chisq2D_test_cont(x, y, Ranges, nbins, minexpcount)
      if(missing(rnull)) outpvals <- TS_cont_pval(x, y)
    } else if(missing(rnull)) {
      outchi <- chisq2D_test_disc(dta, minexpcount)
    }
  }

  TS_data <- calcTS(dta, TS, typeTS, TSextra)
  validate_ts_output(TS_data)

  if(B == 0L) {
    pvals <- rep(NA_real_, length(TS_data)); names(pvals) <- names(TS_data)
    return(.new_MD2sample_test(TS_data, pvals, inp, B_requested, 0L,
                               NA_integer_, typeTS, LargeSampleOnly, call))
  }

  if(!LargeSampleOnly) {
    maxProcessor <- makemaxProcessor(maxProcessor, dta, TS, typeTS, TSextra, B,
                                     SuppressMessages, tmp$useSingleProcessor)
    if(tmp$useSingleProcessor && !SuppressMessages && maxProcessor == 1L)
      message("Parallel processing is not possible if custom TS is written in C++. Switching to single processor")
    B_used <- ceiling(B/maxProcessor)*maxProcessor
    if(maxProcessor == 1L) {
      outTS <- testC(dta, TS, typeTS, TSextra, B=B_used)
    } else {
      cl <- parallel::makeCluster(maxProcessor)
      on.exit(parallel::stopCluster(cl), add=TRUE)
      if(!is.null(seed)) parallel::clusterSetRNGStream(cl, iseed=seed)
      z <- parallel::clusterCall(cl, testC, dta=dta, TS=TS, typeTS=typeTS,
                                 TSextra=TSextra, B=as.integer(B_used/maxProcessor))
      p <- z[[1]]$p.values
      for(i in 2:maxProcessor) p <- p + z[[i]]$p.values
      outTS <- list(statistics=z[[1]]$statistics, p.values=p/maxProcessor)
    }
  } else {
    maxProcessor <- NA_integer_
    B_used <- 0L
    outTS <- list(statistics=NULL, p.values=NULL)
  }

  if(CustomTS) {
    out <- signif_digits(outTS)
  } else {
    s <- c(outTS$statistics, outpvals$statistics, outchi$statistics)
    p <- c(outTS$p.values, outpvals$p.values, outchi$p.values)
    if(doMethods[1] != "all") { s <- s[doMethods]; p <- p[doMethods] }
    out <- signif_digits(list(statistics=s, p.values=p))
  }

  .new_MD2sample_test(out$statistics, out$p.values, inp, B_requested, B_used,
                      maxProcessor, typeTS, LargeSampleOnly, call)
}

.new_MD2sample_test <- function(statistics, p.values, inp, B_requested, B_used,
                                maxProcessor, typeTS, LargeSampleOnly, call) {
  if(length(statistics) != length(p.values))
    stop("statistics and p.values must have the same length", call.=FALSE)
  results <- data.frame(method=names(statistics), statistic=unname(statistics),
                        p.value=unname(p.values), row.names=NULL, check.names=FALSE)
  structure(list(results=results, statistics=statistics, p.values=p.values,
                 metadata=list(data.type=if(inp$Continuous) "continuous" else "discrete",
                               n.x=inp$n.x, n.y=inp$n.y, dimension=inp$Dim,
                               B.requested=B_requested, B.used=B_used,
                               maxProcessor=maxProcessor, typeTS=typeTS,
                               transformed=inp$DoTransform,
                               LargeSampleOnly=LargeSampleOnly),
                 call=call), class="MD2sample_test")
}

#' Methods for MD2sample two-sample test results
#'
#' Print, summarize, and convert structured \code{MD2sample_test} objects.
#'
#' @param x An object of class \code{"MD2sample_test"}, or for
#'   \code{print.summary.MD2sample_test}, an object of class
#'   \code{"summary.MD2sample_test"}.
#' @param object An object of class \code{"MD2sample_test"}.
#' @param ... Further arguments passed to the relevant method.
#' @param row.names Ignored; included for compatibility with \code{as.data.frame}.
#' @param optional Ignored; included for compatibility with \code{as.data.frame}.
#' @return The print methods return their input invisibly; \code{summary()}
#'   returns a summary object; \code{as.data.frame()} returns columns
#'   \code{method}, \code{statistic}, and \code{p.value}.
#' @name MD2sample_test-methods
#' @export
print.MD2sample_test <- function(x, ...) {
  cat("\nMD2sample two-sample tests\n")
  cat("Data:", x$metadata$data.type,
      " | n.x =", x$metadata$n.x,
      " | n.y =", x$metadata$n.y,
      " | dimension =", x$metadata$dimension,
      " | B =", x$metadata$B.used, "\n\n")
  print(x$results, row.names=FALSE, ...)
  invisible(x)
}

#' @rdname MD2sample_test-methods
#' @export
summary.MD2sample_test <- function(object, ...) {
  out <- list(call=object$call, results=object$results, metadata=object$metadata)
  class(out) <- "summary.MD2sample_test"
  out
}

#' @rdname MD2sample_test-methods
#' @export
print.summary.MD2sample_test <- function(x, ...) {
  cat("\nMD2sample two-sample test summary\n\n")
  cat("Data type: ", x$metadata$data.type, "\n", sep="")
  cat("Sample sizes: ", x$metadata$n.x, " and ", x$metadata$n.y, "\n", sep="")
  cat("Dimension: ", x$metadata$dimension, "\n", sep="")
  cat("Simulation runs requested: ", x$metadata$B.requested, "\n", sep="")
  cat("Simulation runs used: ", x$metadata$B.used, "\n", sep="")
  if(!is.na(x$metadata$maxProcessor)) cat("Processors used: ", x$metadata$maxProcessor, "\n", sep="")
  cat("\nTest results:\n")
  print(x$results, row.names=FALSE)
  invisible(x)
}

#' @rdname MD2sample_test-methods
#' @export
as.data.frame.MD2sample_test <- function(x, row.names=NULL, optional=FALSE, ...) x$results
