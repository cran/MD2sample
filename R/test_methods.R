#' This function checks whether the correct methods have been requested
#' @param  doMethods ="all" Which methods should be included?
#' @param  Continuous is data continuous
#' @param  ReturnMethodNames = FALSE should names of methods be returned?
#' @keywords internal
#' @return TRUE/FALSE or a character vector
#' @export 
test_methods=function(doMethods, Continuous, ReturnMethodNames=FALSE) {
    if(doMethods[1]=="all") return(FALSE)
    if(Continuous) methods=
                 c("KS","K","CvM","AD","NN1", "NN5", "AZ","BF","BG",
                   "FR","NN0","CF1","CF2","CF3","CF4",
                   "Ball",  "ES", "EP")
    else methods=c("KS","K","CvM","AD","NN","AZ", "BF","ChiSquare")
    if(ReturnMethodNames) return(methods)
    Good=TRUE
    for(i in seq_along(doMethods)) {
      if(!(doMethods[i]%in%methods)) {Good=FALSE;break}
    }
    if(Good) return(FALSE)
    message(paste0(doMethods[i]," is not an included method for ", 
                   ifelse(Continuous, "continuous", "discrete"), " data!"))
    if(Continuous) {
         message("For continuous data included methods are")
         message("Method               Code")
         message("Kolmogorov-Smirnov   KS")
         message("Kuiper               K")
         message("Cramer-vonMises      CvM")
         message("Anderson-Darling     AD")
         message("1-nearest neighbor   NN1")
         message("5-nearest neighbor   NN5")
         message("Aslan-Zech           AZ")
         message("Baringhaus-Franz     BF")
         message("Biswas-Ghosh         BG")
         message("Friedman-Rafski      FR")
         message("x nearest neighbor   NN0")
         message("Chen-Friedman        CF1-CF4")
         message("Ball Divergence      Ball")
         message("Chi square tests     ES, EP")
    }
    if(!Continuous) {
      message("For discrete data included methods are")
      message("Method               Code")
      message("Kolmogorov-Smirnov   KS")
      message("Kuiper               K")
      message("Cramer-vonMises      CvM")
      message("Anderson-Darling     AD")
      message("Nearest Neigbor      NN")
      message("Aslan-Zech           AZ")
      message("Baringhaus-Franz     BF")
      message("Chi square test      Chisquare")
    }
    TRUE
}
