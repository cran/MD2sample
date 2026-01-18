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
    if(!is.numeric(samplingmethod))  
      samplingmethod=ifelse(samplingmethod=="independence", 1, 2)
# create function rxy which generates data, with two arguments                       
    if(length(list(...))==0) { # f has 0 arguments
       rxy=function(a=0, b=0) f()
       avals=0
       bvals=0
    }
    if(length(list(...))==1) { # f has 1 argument
       rxy=function(a=0, b=0) f(a)
       avals=list(...)[[1]]
       bvals=0
    }
    if(length(list(...))==2) { # f has 2 arguments
       rxy=function(a=0, b=0) f(a,b)
       avals=list(...)[[1]]
       bvals=list(...)[[2]]
    }
  # check that avals and bvals have the same length. 
  # If they do, matrix of powers is returned without row names.
  # If one of them is a scalar, make it the same length as the other and use those
  # values as row names
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
# generate one data set as an example, do some setup 
  dta = rxy(avals[1], bvals[1])
  Continuous=TRUE
  if(is.matrix(dta)) { #Discrete Data
    dta=list(x=dta[,3], y=dta[,4], vals_x=dta[,1], vals_y=dta[,2])
    x=matrix(1:4,2,2) #just some dummy numbers
    y=matrix(1:4,2,2)
    Continuous=FALSE
    DoTransform=FALSE
  }
  if(test_methods(doMethods, Continuous)) return(NULL)
  if(Continuous) {
    x=dta$x
    y=dta$y
    Dim=ncol(x) # dimension of data
    if(nrow(y)<nrow(x)) { 
      if(!SuppressMessages) message("sample size of x should not be larger than sample size of y")
      return(NULL)
    }
  }  
  if(DoTransform) {
    dta=transform01(dta)
    x=dta$x
    y=dta$y
    Ranges=matrix(c(0, 1, 0, 1),2,2)
  }
  if(missing(TSextra)) TSextra=list(aaa=0)
  if(Continuous)
     TSextra = c(TSextra, 
               knn=function(x) FNN::get.knn(x, 5)$nn.index,
               dist=function(dta) find_dist(dta),
               distances=list(find_dist(list(x=x,y=y))),
               DoTransform=DoTransform,
               ParametricBootstrap=FALSE)
  else TSextra = c(TSextra, 
               dist=function(dta) NULL,
               organize=function(dta) dta=dta[order(dta[,1], dta[,2]), ],
               samplingmethod=samplingmethod,
               ParametricBootstrap=FALSE)
  if(!missing(rnull)) {
     TSextra$ParametricBootstrap=TRUE
     TSextra=c(TSextra, rnull=rnull, rawdta=list(dta))
  }   
  pwrchi=NULL # no chi square test
  if(missing(TS)) { # do included methods
    CustomTS=FALSE
    if(Continuous) { # Continuous Data
        typeTS=1
        TS=TS_cont
    }
    else { # Discrete Data
        typeTS=4
        TS=TS_disc
    }  
  }
  else { # do user-supplied tests
    CustomTS=TRUE
    if(Continuous) typeTS=length(formals(TS))
    else typeTS=ifelse(length(formals(TS))==5, 6, 5)
  }
  TS_data=calcTS(dta, TS, typeTS, TSextra)
  if(is.null(names(TS_data))) {
    if(!SuppressMessages) message("output of TS routine has to be a named vector!")
    return(NULL)
  }  
  methodnames=names(TS_data)
  
# Do a time check for power
  if(With.p.value) maxProcessor=1
  if(missing(maxProcessor)) {
    ncores=max(parallel::detectCores(logical=FALSE)-1,1)
    tm=timecheck(dta, TS, typeTS, TSextra)
    if(length(tm)==1) tm=c(tm,0)
    totaltime=2*tm*length(avals)*B
    if(max(totaltime)<20 | B<=2*ncores) 
       if(!SuppressMessages) message("maxProcessor set to 1 for faster computation")
    else if(!SuppressMessages) message(paste("Using ", ncores," cores..")) 
    maxProcessor1=1
    if(totaltime[1]>20 & B>2*ncores) 
       maxProcessor1=ncores
    maxProcessor2=1
    if(typeTS==1 & totaltime[2]>20 & B>2*ncores) 
       maxProcessor2=ncores
    if(max(totaltime)>20 & B>2*ncores) {
       if(maxProcessor1==1 & maxProcessor2>1)
          totaltime=30+totaltime[2]/maxProcessor2
       if(maxProcessor1>1 & maxProcessor2==1)
          totaltime=30+totaltime[1]/maxProcessor1
       if(maxProcessor1>1 & maxProcessor2>1)
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
    maxProcessor1=maxProcessor
    maxProcessor2=maxProcessor
  }
# Run power routine for new test which returns p value(s)
  if(With.p.value) {
    pwr=power_pvals(rxy, avals, bvals, TS=TS, typeTS, TSextra, alpha=alpha, B=B)
    if(length(list(...))==0) rownames(pwr)=NULL
    if(length(list(...))==1) rownames(pwr)=avals
    if(length(list(...))==2) rownames(pwr)=paste0(avals,"|",bvals)  
    if(doMethods[1]!="all") pwr=pwr[, doMethods, drop=FALSE]
    return(round(pwr, 4))
  }

# Run power routines for continuous data
  if(!LargeSampleOnly) {
    if(maxProcessor1==1) {
      tmp=powerC(rxy, avals, bvals, TS, typeTS, TSextra, B)
      Data=tmp$Data
      Simulated=tmp$Simulated
      paramalt=tmp$paramalt
    }    
    else {
      cl=parallel::makeCluster(maxProcessor1)
      z=parallel::clusterCall(cl, powerC, 
                              rxy,  avals, bvals,
                              TS, typeTS, TSextra, round(B[1]/maxProcessor1))
      parallel::stopCluster(cl)
      Simulated=z[[1]][["Simulated"]]
      Data=z[[1]][["Data"]]
      paramalt=z[[1]][["paramalt"]]
      for(i in 2:maxProcessor1) {
        Simulated=rbind(Simulated, z[[i]][["Simulated"]])
        Data=rbind(Data, z[[i]][["Data"]])
        paramalt=rbind(paramalt,z[[i]][["paramalt"]])
      }  
    }
    pwr=matrix(0, length(avals), length(TS_data))
    colnames(pwr)=names(TS_data)
    for(i in seq_along(avals)) {
      Index=c(1:nrow(Data))[paramalt[,1]==avals[i]&paramalt[,2]==bvals[i]]
      tmpD=Data[Index, , drop=FALSE]
      tmpS=Simulated[Index, , drop=FALSE]
      crtval=apply(tmpS, 2, quantile, prob=1-alpha, na.rm=TRUE)
      for(j in seq_along(crtval)) 
        pwr[i, j]=sum(tmpD[ ,j]>crtval[j])/nrow(tmpD)
    }
  }  
  pwrothers=NULL
  if(missing(rnull) & (typeTS %in% c(1, 4))) {
     if(maxProcessor2==1) {
          pwrothers=power_pvals(rxy, avals, bvals,  
                              TS=TS, typeTS=typeTS, TSextra=TSextra,
                              nbins=nbins, minexpcount=minexpcount, 
                              Ranges=Ranges, alpha=alpha, B=B)
      }  
      else { 
         cl <- parallel::makeCluster(maxProcessor2)
         u = parallel::clusterCall(cl, power_pvals, 
            rxy, avals, bvals,
            TS=TS, typeTS=typeTS, TSextra=TSextra,
            nbins=nbins, minexpcount=minexpcount, Ranges=Ranges,
            alpha=alpha, B=round(B/maxProcessor2))
        parallel::stopCluster(cl)  
        # Average power over cores  
        pwrothers=u[[1]]
        for(i in 2:maxProcessor2) pwrothers=pwrothers+u[[i]]
        pwrothers = pwrothers/maxProcessor2
      }
  }
  if(LargeSampleOnly) pwr=pwrothers
  else if(!CustomTS) pwr = cbind(pwr, pwrothers)
  if(length(list(...))==0) rownames(pwr)=NULL
  if(length(list(...))==1) rownames(pwr)=avals
  if(length(list(...))==2) rownames(pwr)=paste0(avals,"|",bvals)
  if(doMethods[1]!="all") pwr=pwr[, doMethods, drop=FALSE]
  round(pwr, 4)
}
