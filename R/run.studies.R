#' Benchmarking for Multivariate Two-Sample Tests
#' 
#' This function runs the case studies included in the package.
#' 
#' For details consult vignette(package="MD2sample")
#' 
#' @param Continuous =TRUE, run cases for continuous data.
#' @param study either the name of the study, or its number in the list. If missing all the studies are run.
#' @param TS routine to calculate new test statistics. 
#' @param TSextra list passed to TS (optional).
#' @param With.p.value =FALSE, does user supplied routine return p values?
#' @param nsample = 200, desired sample size. 200 is used in included case studies.
#' @param alpha =0.05,  type I error probability of tests. 0.05 is used in included case studies.
#' @param param_alt (list of) values of parameter under the alternative hypothesis. 
#'                  If missing included values are used.
#' @param SuppressMessages =FALSE, should informative messages be printed?
#' @param B = 1000, number of simulation runs.
#' @param  maxProcessor  number of cores to use. If missing the number of physical cores-1 
#'             is used. If set to 1 no parallel processing is done.
#' @return A (list of ) matrices of p.values.
#' @examples
#' #The new test is a (included) chi square test:
#' TSextra=list(which="pval", nbins=rbind(c(3,3), c(4,4)))
#' run.studies(Continuous=TRUE, study=c("NormalD2", "tD2"), 
#'           TS=MD2sample::chiTS.cont, TSextra=TSextra, B=100)
#' @export
run.studies <- function(Continuous=TRUE, study, TS, TSextra, With.p.value=FALSE,  
          nsample=200, alpha=0.05, param_alt, 
          SuppressMessages =FALSE, B=1000, maxProcessor) {
  list.of.studies=MD2sample::case.studies(ReturnCaseNames=TRUE)
  if(missing(study)) study=1:length(list.of.studies)
  if(is.numeric(study)) study=list.of.studies[study]
  if(!Continuous) study=study[!endsWith(study,"D5")]
  RerunIncludedTests=FALSE
  if(missing(TS)) {
      RerunIncludedTests=TRUE
      if(!missing(param_alt)) {
          if(!SuppressMessages) 
            message("if TS is missing included tests are rerun for new parameters")
      }
  }    
  if(RerunIncludedTests) {
      if(length(study)>1) {
        if(is.matrix(param_alt)) {
           if(nrow(param_alt)!=length(study))
             if(!SuppressMessages) 
               message("param_alt should be a matrix with the one row for each study")
          
        }
        else {
          if(!SuppressMessages) message("if more than one study is run 
      param_alt has to be a matrix with the one row for each study")
          return(NULL)
        }  
      }
      else param_alt=rbind(param_alt)
      out=as.list(1:length(study))
      names(out)=study
      for(i in seq_along(study)) {
        tmp=MD2sample::case.studies(study[i])
        if(!Continuous) 
           tmp=MD2sample::case.studies(study[i], nbins=tmp$nbins)
        if(!SuppressMessages) message(paste("Running case study", study[i],"..."))
        out[[i]]=MD2sample::twosample_power(tmp$f, param_alt[i,], 
                      alpha=alpha,  SuppressMessages=SuppressMessages, 
                      B=B, maxProcessor=maxProcessor) 
      }  
      if(length(out)==1) return(out[[1]])
      return(out)
  } 
  if(Continuous) typeTS=length(formals(TS))
  else typeTS=ifelse(length(formals(TS))==3, 6, 5)  
  if(Continuous) pwr=MD2sample::power_studies_results[["Alt Cont"]]
  else pwr=MD2sample::power_studies_results[["Alt Disc"]]
  nstudy=length(study)
  for(i in seq_along(study)) {
    tmp=case.studies(study[i], nsample)
    if(!Continuous) 
      tmp=case.studies(study[i], nbins=tmp$nbins)
    if(!SuppressMessages) message(paste("Running case study", study[i],
                                       "with param_alt=", tmp$param_alt[2]))
    rxy=function(a,b=0) tmp$f(a)
    if(With.p.value) {
        if(missing(TSextra)) TSextra=list(aaa=0)
        pwrTS=MD2sample::power_pvals(rxy, tmp$param_alt[2], 
            0, TS, typeTS, TSextra, alpha=alpha, B=B)
    }    
    else  pwrTS=MD2sample::twosample_power(tmp$f, tmp$param_alt[2], 
                                   TS=TS, TSextra=TSextra, alpha=alpha,  
                                   SuppressMessages=SuppressMessages,
                                   B=B, maxProcessor=maxProcessor)   
    if(i==1) {
      allpwr=pwr[study, , drop=FALSE]
      for(m in 1:length(pwrTS)) allpwr=cbind(0, allpwr)
    }  
    allpwr[study[i], 1:length(pwrTS)]=pwrTS
  }
  colnames(allpwr) = c(colnames(pwrTS), colnames(pwr))
  if(nstudy>1) {
    a1=apply(allpwr, 1, rank)
    names(a1)=c(colnames(pwrTS), colnames(pwr))
    message("Average number of times a test is close to best:")
    print(sort(apply(a1,1,mean)))
  }  
  allpwr
}
