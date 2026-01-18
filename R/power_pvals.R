#' This function finds the power for tests that find their own p values
#' @param  rxy  a function to generate data
#' @param  avals vector of parameter values
#' @param  bvals vector of parameter values
#' @param  TS routine that runs the test and returns p values
#' @param  typeTS  type of problem, continuous or discrete
#' @param  TSextra =list(a=0) a list of things passed to TS, if needed
#' @param  nbins =c(5, 5), bins for chi square tests.
#' @param  minexpcount =5, lowest required count for chi-square test
#' @param  Ranges =matrix(c(-Inf, Inf, -Inf, Inf),2,2) a 2x2 matrix with lower and upper bounds
#' @param  alpha =0.05 type I error probability of test
#' @param  B =1000 number of simulation runs
#' @keywords internal
#' @return A matrix of power values
#' @export
power_pvals = function(rxy, avals, bvals, TS, typeTS, TSextra=list(a=0),
                       nbins=c(5,5), minexpcount=5, 
                       Ranges =matrix(c(-Inf, Inf, -Inf, Inf),2,2),
                       alpha=0.05, B=1000) {
  
   dta = rxy(avals[1], bvals[1])
   Continuous=TRUE
   if(is.matrix(dta)) Continuous=FALSE
   DoTransform=FALSE
   if(Continuous) {
     if("DoTransform" %in% names(TSextra)) DoTransform=TSextra$DoTransform
     else {
       z=c(dta$x, dta$y)
       if(min(z)<0 | max(z)>1) DoTransform=TRUE
     }
     if(DoTransform) {
       dta = transform01(dta)
       Ranges =matrix(c(0, 1, 0, 1),2,2)
     }   
     Dim = ncol(dta$x)
     DoChi=FALSE
     if(typeTS==1 & Dim==2) DoChi=TRUE
     if(typeTS==1) {
       typeTS=2 
       TS=function(x,y) TS_cont_pval(x, y)$p.values
     }   
     TS_data = calcTS(dta, TS, typeTS, TSextra)
     numtests=length(TS_data)
     if(typeTS<=3) Continuous = TRUE
     else Continuous = FALSE
     pwr=matrix(0, length(avals), numtests+ifelse(DoChi, 2, 0))
     if(DoChi) colnames(pwr) = c(names(TS_data), "ES", "EP")
     else colnames(pwr) = names(TS_data)
     rownames(pwr)=avals
     for(i in 1:length(avals)) {
       NoGood=NULL
       for(j in 1:B) {
         dta=rxy(avals[i], bvals[i])
         if(DoTransform) dta=transform01(dta)
         TS_sim = calcTS(dta, TS, typeTS, TSextra)
         if(any(is.nan(TS_sim))) {j=j-1;next}
         if(any(TS_sim<0)) {
             NoGood=c(NoGood, seq_along(TS_sim)[TS_sim<0])
         }
         pwr[i, 1:numtests] = pwr[i, 1:numtests] + ifelse(TS_sim<alpha, 1, 0)
         if(DoChi) {
           if(length(nbins)==1) nbins=c(nbins, nbins)
           chi=chisq2D_test_cont(dta$x, dta$y, Ranges, nbins, minexpcount,
                                 SuppressMessages=TRUE)$p.values
           pwr[i, numtests+1:2] = pwr[i, numtests+1:2] + ifelse(chi<alpha, 1, 0)
         }   
       }
       pwr[i, ] = pwr[i, ,drop=FALSE]/B
       if(length(NoGood)>0) pwr[i, NoGood]=NA
     }
   }   
   else {#Discrete Data
     if(typeTS==4) {      
        pwr=matrix(0,  length(avals), 1)
        colnames(pwr)="Chisquare"
        rownames(pwr)=avals
        for(i in 1:length(avals)) {
          for(j in 1:B) {
            dta=rxy(avals[i], bvals[i])
            chi=chisq2D_test_disc(dta, minexpcount)$p.values
            pwr[i,1] = pwr[i,1] + ifelse(chi<alpha, 1, 0)
          }
          pwr[i,1] = pwr[i,1]/B          
        }
     }
     else {      
       dta=rxy(avals[1], bvals[1])
       if(typeTS>=4) {
          dn=colnames(dta)
          if("x"%in%dn & "y"%in%dn & "vals_x"%in%dn & "vals_y"%in%dn)
             dta=list(x=dta[,"x"],y=dta[,"y"],vals_x=dta[,"vals_x"],vals_y=dta[,"vals_y"])
          else {
            dta=list(x=dta[,3],y=dta[,4],vals_x=dta[,1],vals_y=dta[,2])
            message("data matrix is missing column names. Assumed order is:")
            message("vals_x, vals_y, x, y")
          }
       }
       TS_sim=calcTS(dta, TS, typeTS, TSextra)
       pwr=matrix(0,  length(avals), length(TS_sim))
       colnames(pwr)=names(TS_sim)
       rownames(pwr)=avals
       for(i in 1:length(avals)) {
         for(j in 1:B) {
           dta=rxy(avals[i], bvals[i])
           if(typeTS>=4) {
             if("x"%in%dn & "y"%in%dn & "vals_x"%in%dn & "vals_y"%in%dn)
               dta=list(x=dta[,"x"],y=dta[,"y"],vals_x=dta[,"vals_x"],vals_y=dta[,"vals_y"])
             else {
               dta=list(x=dta[,3],y=dta[,4],vals_x=dta[,1],vals_y=dta[,2])
             }
           }
           TS_sim=calcTS(dta, TS, typeTS, TSextra)
           for(k in 1:ncol(pwr)) 
              pwr[i,k] = pwr[i,k] + ifelse(TS_sim[k]<alpha, 1, 0)/B
         }
       }
     }
   }
   pwr
}
