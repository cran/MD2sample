#' Check requested two-sample methods
#' @param doMethods Methods requested, or "all".
#' @param Continuous Is the data continuous?
#' @param ReturnMethodNames Return available method names instead of validating?
#' @return Invisibly FALSE after successful validation, or the available names.
#' @keywords internal
test_methods <- function(doMethods, Continuous, ReturnMethodNames=FALSE) {
  methods <- if(Continuous) {
    c("KS","K","CvM","AD","NN1","NN5","AZ","BF","BG","MMD",
      "FR","NN0","CF1","CF2","CF3","CF4","Ball","ES","EP")
  } else {
    c("KS","K","CvM","AD","NN","AZ","BF","ChiSquare")
  }
  if(ReturnMethodNames) return(methods)
  if(length(doMethods) < 1L || anyNA(doMethods))
    stop("doMethods must contain at least one method name.", call.=FALSE)
  if(length(doMethods)==1L && identical(doMethods[1L], "all")) return(invisible(FALSE))
  bad <- setdiff(doMethods, methods)
  if(length(bad))
    stop("Unknown method(s) for ", if(Continuous) "continuous" else "discrete",
         " data: ", paste(bad, collapse=", "), ". Available methods are: ",
         paste(methods, collapse=", "), ".", call.=FALSE)
  invisible(FALSE)
}
