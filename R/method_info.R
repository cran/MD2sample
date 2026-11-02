#' Information about the included methods
#'
#' Returns a table with one row for each method included in MD2sample: the code
#' used in the output of \code{twosample_test()} and \code{twosample_power()}, its
#' full name, the type of data, how p values are found, a reference, and notes on
#' methods that are identical to others.
#'
#' Some methods are the same test under different names: FR and CF1 are both the
#' original edge-count test of Friedman and Rafsky on the minimum spanning tree,
#' and CF3, the weighted edge-count test of Chen and Friedman, is identical to CF1
#' if the two data sets have the same sample size. The codes CF1 to CF4 follow the
#' order in which \code{gTests::g.tests} returns the tests: original, generalized,
#' weighted and max-type edge-count test.
#'
#' @param Continuous =NULL, if TRUE or FALSE only the methods for continuous or
#'        discrete data are returned.
#' @return A data frame with columns \code{code}, \code{data}, \code{name},
#'         \code{p.value}, \code{reference} and \code{note}.
#' @examples
#' method_info(Continuous=TRUE)
#' @export
method_info <- function(Continuous=NULL) {
  m <- function(code, data, name, p.value, reference, note="")
    data.frame(code=code, data=data, name=name, p.value=p.value,
               reference=reference, note=note)
  sim <- "simulation"
  ls <- "large sample theory"
  out <- rbind(
    m("KS", "continuous", "Kolmogorov-Smirnov", sim, "Kolmogorov (1933), Smirnov (1939)",
      "empirical distribution functions of the lower-left orthant at the data points"),
    m("K", "continuous", "Kuiper", sim, "Kuiper (1960)"),
    m("CvM", "continuous", "Cramer-von Mises", sim, "Anderson (1962)"),
    m("AD", "continuous", "Anderson-Darling", sim, "Anderson and Darling (1952)"),
    m("NN1", "continuous", "Nearest neighbors, 1 neighbor", sim, "Schilling (1986)"),
    m("NN5", "continuous", "Nearest neighbors, 5 neighbors", sim, "Schilling (1986)"),
    m("AZ", "continuous", "Aslan-Zech energy test", sim, "Aslan and Zech (2005)"),
    m("BF", "continuous", "Baringhaus-Franz", sim, "Baringhaus and Franz (2004)"),
    m("BG", "continuous", "Biswas-Ghosh", sim, "Biswas and Ghosh (2014)"),
    m("MMD", "continuous", "Maximum mean discrepancy, aggregated over bandwidths", sim,
      "Gretton et al. (2012), Schrab et al. (2023)"),
    m("FR", "continuous", "Friedman-Rafsky (original edge-count test)", ls,
      "Friedman and Rafsky (1979)", "same test as CF1"),
    m("NN0", "continuous", "Simple nearest neighbor test", ls, "Schilling (1986)"),
    m("CF1", "continuous", "Original edge-count test", ls, "Chen and Friedman (2017)",
      "same test as FR"),
    m("CF2", "continuous", "Generalized edge-count test", ls, "Chen and Friedman (2017)"),
    m("CF3", "continuous", "Weighted edge-count test", ls, "Chen and Friedman (2017)",
      "identical to CF1 if both samples have the same size"),
    m("CF4", "continuous", "Max-type edge-count test", ls, "Chen and Friedman (2017)"),
    m("Ball", "continuous", "Ball divergence", ls, "Pan et al. (2018)"),
    m("ES", "continuous", "Chi-square test, equally spaced bins", ls, "",
      "two-dimensional data only"),
    m("EP", "continuous", "Chi-square test, equal probability bins", ls, "",
      "two-dimensional data only"),
    m("KS", "discrete", "Kolmogorov-Smirnov", sim, "Kolmogorov (1933), Smirnov (1939)"),
    m("K", "discrete", "Kuiper", sim, "Kuiper (1960)"),
    m("CvM", "discrete", "Cramer-von Mises", sim, "Anderson (1962)"),
    m("AD", "discrete", "Anderson-Darling", sim, "Anderson and Darling (1952)"),
    m("NN", "discrete", "Nearest neighbors (adjacent cells)", sim, ""),
    m("AZ", "discrete", "Aslan-Zech energy test", sim, "Aslan and Zech (2005)"),
    m("BF", "discrete", "Baringhaus-Franz", sim, "Baringhaus and Franz (2004)"),
    m("ChiSquare", "discrete", "Chi-square test", ls, "",
      "called Chisquare in the output of twosample_power() and in power_studies_results")
  )
  if(!is.null(Continuous))
    out <- out[out$data == if(isTRUE(Continuous)) "continuous" else "discrete", ]
  rownames(out) <- NULL
  out
}
