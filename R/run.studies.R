#' Benchmarking for Multivariate Two-Sample Tests
#' 
#' This function runs the case studies included in the package.
#' 
#' For details consult vignette(package="MD2sample")
#' 
#' @param study either the name of the study, or its number in the list. If missing all the studies are run.
#' @param Continuous =TRUE, run cases for continuous data.
#' @param TS routine to calculate new test statistics. 
#' @param TSextra list passed to TS (optional).
#' @param With.p.value =FALSE, does user supplied routine return p values?
#' @param alpha =0.05,  type I error probability of tests. 0.05 is used in included case studies.
#' @param param_alt vector or matrix of values of parameters under the alternative hypothesis. 
#'                  If missing included values are used.
#' @param nsample =c(200, 200), sample sizes for x and y data sets.                
#' @param SuppressMessages =FALSE, should informative messages be printed?
#' @param B = 1000, number of simulation runs.
#' @param  maxProcessor  number of cores to use. If missing the number of physical cores-1 
#'             is used. If set to 1 no parallel processing is done.
#' @return A (list of ) matrices of p.values.
#' @examples
#' #The new test is a (included) chi square test:
#' TSextra=list(which="pval", nbins=rbind(c(3,3), c(4,4)))
#' run.studies(study=c("NormalD2", "tD2"), Continuous=TRUE,  
#'           TS=MD2sample::chiTS.cont, TSextra=TSextra, 
#'           With.p.value=TRUE, B=100)
#' @export
run.studies <- function(study, Continuous=TRUE, TS, TSextra, With.p.value=FALSE,  
                        alpha=0.05, param_alt, nsample=c(200, 200), 
                        SuppressMessages =FALSE, B=1000, maxProcessor) {
  
  # Get all available case-study names from the package.
  list.of.studies=MD2sample::case.studies(ReturnCaseNames=TRUE)
  
  # If no study is specified, run all available studies.
  if(missing(study)) study=1:length(list.of.studies)
  
  # Allow studies to be selected by numeric index.
  if(is.numeric(study)) study=list.of.studies[study]
  
  # For discrete runs, remove studies ending in "D5".
  if(!Continuous) study=study[!endsWith(study,"D5")]
  
  # If only one sample size is supplied, use it for both samples.
  if(length(nsample)==1) nsample=c(nsample, nsample)
  
  # Track whether results differ from the package's stored default studies.
  if(length(nsample)==1) nsample=c(nsample, nsample)
  ChangedStudies=FALSE 
  if(alpha!=0.05 || nsample[1]!=200 || nsample[2]!=200)
    ChangedStudies=TRUE
  
  # If alternative parameters are not supplied, get them from case.studies().
  if(missing(param_alt)) {
    param_alt=cbind(1:length(study))
    rownames(param_alt)=study
    
    for(i in seq_along(study)) {
      tmp=MD2sample::case.studies(study[i])
      param_alt[i,1]=tmp$param_alt[2]
    }
  }
  else {
    # Ensure vector input becomes a one-column matrix.
    if(is.vector(param_alt)) param_alt=cbind(param_alt)
  }
  
  # Check whether the supplied alternative parameters match package defaults.
  if(ncol(param_alt)==1) {
    for(i in seq_along(study)) { 
      tmp=MD2sample::case.studies(study[i])
      
      if(any(param_alt[i, 1]!=tmp$param_alt[2])) 
        ChangedStudies=TRUE
    }
  }  
  else ChangedStudies=TRUE
  
  # If no custom test statistic is supplied, rerun included package tests.
  RerunIncludedTests=ifelse(missing(TS), TRUE, FALSE)
  
  # For a custom test statistic, determine the calling convention.
  if(!missing(TS)) {
    if(Continuous) typeTS=length(formals(TS))
    else typeTS=ifelse(length(formals(TS))==3, 6, 5)  
  }  
  
  # Initialize list to store power results for each study.
  newpwr=as.list(study)
  names(newpwr)=study
  
  for(i in seq_along(study)) {
    
    # Retrieve generator and study settings with requested sample sizes.
    tmp=MD2sample::case.studies(study[i], nx=nsample[1], ny=nsample[2])
    
    # For discrete studies, preserve the study-specific number of bins.
    if(!Continuous) 
      newpwr[[i]]=MD2sample::case.studies(study[i], nx=nsample[1], ny=nsample[2],
                                          nbins=tmp$nbins)
    
    if(!SuppressMessages) message(paste("Running case study", study[i],"..."))
    
    # Run power for built-in tests.
    if(RerunIncludedTests) {
      newpwr[[i]]=MD2sample::twosample_power(tmp$f, param_alt[i,], 
                                             alpha=alpha,  SuppressMessages=SuppressMessages, 
                                             B=B, maxProcessor=maxProcessor) 
    }    
    
    # Run power for a custom test that returns p-values directly.
    if(!RerunIncludedTests & With.p.value) {
      if(missing(TSextra)) TSextra=list(aaa=0)
      
      # Wrap the study generator so it has the argument structure expected
      # by power_pvals().
      rxy=function(a,b=0) tmp$f(a)
      
      newpwr[[i]]=MD2sample::power_pvals(rxy, param_alt[i, ], 
                                         0, TS, typeTS, TSextra, alpha=alpha, B=B)
    }    
    
    # Run power for a custom test statistic using simulated critical values.
    if(!RerunIncludedTests & !With.p.value) {
      newpwr[[i]]=MD2sample::twosample_power(tmp$f, param_alt[i, ], 
                                             TS=TS, TSextra=TSextra, alpha=alpha,  
                                             SuppressMessages=SuppressMessages,
                                             B=B, maxProcessor=maxProcessor)   
    }    
  } 

  # If settings differ from defaults, no comparison to stored results is made.
  if(ChangedStudies) {
    if(length(newpwr)==1) newpwr=newpwr[[1]]
    return(newpwr)
  }
  
  # Retrieve stored benchmark power results from the package.
  oldpwr=MD2sample::power_studies_results
  
  if(Continuous) oldpwr=oldpwr[["Alt Cont"]][study, , drop=FALSE]
  else oldpwr=oldpwr[["Alt Disc"]][study, , drop=FALSE]
  
  # Convert the new results list into a matrix.
  pwr=matrix(0, length(study), ncol(newpwr[[1]]))
  
  for(i in seq_along(study)) pwr[i, ]=newpwr[[i]]
  
  dimnames(pwr)=list(study, colnames(newpwr[[1]]))
  
  # For included tests, return only the newly computed power matrix.
  if(RerunIncludedTests) return(pwr)
  
  # For custom tests, append stored package benchmark results.
  allpwr=cbind(pwr, oldpwr)
  
  # If several studies were run, summarize average rankings.
  if(length(study)>1) {
    a1=apply(allpwr, 1, rank)
    names(a1)=c(colnames(newpwr[[1]]), colnames(oldpwr))
    
    message("Average number of times a test is close to best:")
    print(sort(apply(a1,1,mean)))
  }  
  
  allpwr
}
