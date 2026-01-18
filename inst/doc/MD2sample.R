## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(MD2sample)
B=200 

## -----------------------------------------------------------------------------
set.seed(123)

## -----------------------------------------------------------------------------
x1 = mvtnorm::rmvnorm(100, c(0,0))
y1 = mvtnorm::rmvnorm(120, c(0,0))
twosample_test(x1, y1, B=B, maxProcessor = 1)

## -----------------------------------------------------------------------------
TSextra=list(which="statistic", nbins=cbind(c(3,3), c(3,4)))

## -----------------------------------------------------------------------------
twosample_test(x1, y1, TS=chiTS.cont, TSextra=TSextra, 
               B=B, maxProcessor = 1)

## -----------------------------------------------------------------------------
twosample_test_adjusted_pvalue(x1, y1, B=c(B,B),
                               maxProcessor = 1)

## -----------------------------------------------------------------------------
f=function(a=0) {
   S=diag(2) 
   x=mvtnorm::rmvnorm(100, sigma = S)
   S[1,2]=a
   S[2,1]=a
   y=mvtnorm::rmvnorm(120, sigma = S)
   list(x=x, y=y)
}   

## -----------------------------------------------------------------------------
twosample_power(f, c(0,0.5), B=B, maxProcessor=1)

## -----------------------------------------------------------------------------
twosample_power(f, c(0,0.5), TS=chiTS.cont, 
                TSextra=TSextra, B=B, maxProcessor=1)

## -----------------------------------------------------------------------------
f1=function(a=0, n=100) {
   S=diag(2) 
   x=mvtnorm::rmvnorm(100, sigma = S)
   S[1,2]=a
   S[2,1]=a
   y=mvtnorm::rmvnorm(n, sigma = S)
   list(x=x, y=y)
}   

## -----------------------------------------------------------------------------
TSextra=list(which="pvalue", nbins=c(5,5))
twosample_power(f1, c(0,0.5), c(100, 120), 
                TS=chiTS.cont, TSextra=TSextra, B=B,
                With.p.value=TRUE, maxProcessor=1)

## -----------------------------------------------------------------------------
vals_x = 0:5 #possible values of first random variable
vals_y = 0:6 #possible values of second random variable
a1x = rbinom(1000, 5, 0.5) 
a2x = rbinom(1000, 6, 0.5)
a1y = rbinom(1200, 5, 0.5) 
a2y = rbinom(1200, 6, 0.5)
x2=matrix(0, 6*7, 4)
colnames(x2)=c("vals_x", "vals_y", "x", "y")
x2[, 1]=rep(vals_x, length(vals_y))
x2[, 2]=rep(vals_y, each=length(vals_x))
for(i in 0:5) {
  for(j in 0:6) {
    x2[x2[,1]==i&x2[,2]==j, 3]=sum(a1x==i&a2x==j)
    x2[x2[,1]==i&x2[,2]==j, 4]=sum(a1y==i&a2y==j)
  }
}
twosample_test(x2, B=B, maxProcessor = 1)

## -----------------------------------------------------------------------------
twosample_test_adjusted_pvalue(x2, B=c(B,B), maxProcessor=1)

## -----------------------------------------------------------------------------
TSextra=list(which="statistic")
twosample_test(x2, TS=chiTS.disc, TSextra=TSextra,
               B=B, maxProcessor = 1)

## -----------------------------------------------------------------------------
g=function(a=0) {
   x=cbind(rbinom(1000, 5, 0.5), rpois(1000, 1))
   x[,2][x[,2]>5]=5
   lambda=1+a*x[,1]/5
   y=cbind(rbinom(1200, 5, 0.5), rpois(1200, lambda))
   y[,2][y[,2]>5]=5
   vx=0:5
   vy=0:5
   A=matrix(0,length(vx)*length(vy),4)                  
   k=0
   for(i in seq_along(vx))
     for(j in seq_along(vy)) {
       k=k+1
       A[k,1]=vx[i]
       A[k,2]=vy[j]
       A[k,3]=sum(x[,1]==vx[i] & x[,2]==vy[j])
       A[k,4]=sum(y[,1]==vx[i] & y[,2]==vy[j])
     }
   colnames(A)=c("vals_x", "vals_y", "x", "y")
   A
}

## -----------------------------------------------------------------------------
twosample_power(g, c(0, 0.25, 0.5), B=200, maxProcessor=1)

## -----------------------------------------------------------------------------
TSextra=list(which="statistic")
twosample_power(g, c(0, 0.25, 0.5), B=200, 
                TS=chiTS.disc, TSextra=TSextra, maxProcessor=1)
TSextra=list(which="pvalue")
twosample_power(g, c(0, 0.25, 0.5), B=200, 
    TS=chiTS.disc, TSextra=TSextra, With.p.value = TRUE,
    maxProcessor=1)

## -----------------------------------------------------------------------------
# generate real and MC data sets:
f=function(mu) {
    x=mvtnorm::rmvnorm(100, c(mu, mu))
    y=mvtnorm::rmvnorm(100, apply(x, 2, mean), cor(x))
    list(x=x, y=y)
}
#True data is a mixture of normal and uniform
g=function(alpha=0) {
    x=rbind(mvtnorm::rmvnorm((1-alpha)*100, c(0, 0)),
            matrix(runif(200*alpha),ncol=2))
    y=mvtnorm::rmvnorm(100, apply(x, 2, mean), cor(x))
    list(x=x, y=y)
}
# generate two-sample data set
rnull=function(dta) {
   x=mvtnorm::rmvnorm(nrow(dta$x), apply(dta$x, 2, mean), cor(dta$x))
   y=mvtnorm::rmvnorm(nrow(x), apply(x, 2, mean), cor(x))
   list(x=x, y=y)
}

## -----------------------------------------------------------------------------
# Only run these methods for hybrid problem
mt=c("KS", "K", "CvM", "AD", "NN1", "NN5", "AZ", "BF", "BG")
# Null hypothesis is true:
twosample_power(f, c(0, 1), doMethods = mt, B=200, maxProcessor = 1)
twosample_power(f, c(0, 1), rnull=rnull, B=200, maxProcessor = 1)
# Null hypothesis is false:
twosample_power(g, c(0, 0.5), doMethods = mt, B=200, maxProcessor = 1)
twosample_power(g, c(0, 0.5), rnull=rnull, B=200, maxProcessor = 1)

## ----eval=FALSE---------------------------------------------------------------
# run.studies(Continuous=TRUE,
#             TS=newTest_1,
#             study=c("NormalD2", "tD2"),
#             B=1000)

## ----eval=FALSE---------------------------------------------------------------
# run.studies(Continuous=TRUE,
#             study=c("NormalD2", "tD2"),
#             param_alt=cbind(c(0.4, 0.4), c(0.7, 0.7)),
#             alpha=0.1, B=100)

## -----------------------------------------------------------------------------
TSextra=list(which="pvalue", nbins=cbind(c(3,3), c(4,4)))
run.studies(Continuous=TRUE, 
                study=c("NormalD2", "tD2"),
                TS=chiTS.cont, 
                TSextra=TSextra,
                With.p.value = TRUE, 
                B=500,
                SuppressMessages = TRUE,
                maxProcessor=1)

## -----------------------------------------------------------------------------
f=function(a) {
    x=mvtnorm::rmvnorm(500, c(0.5, 0.5))
    y=rbind(matrix(runif(a*1000), ncol=2),
            mvtnorm::rmvnorm((1-a)*500, c(0.5,0.5)))
    list(x=x, y=y)
}    
rnull=function(dta) {
  muhat=apply(dta$x, 2, mean)
  sigmahat=cor(dta$x)
  list(x=mvtnorm::rmvnorm(nrow(dta$x), muhat, sigmahat),  
       y=mvtnorm::rmvnorm(nrow(dta$y), muhat, sigmahat))
}

## -----------------------------------------------------------------------------
dta=f(0) # Null hypothesis is true
twosample_test(dta, rnull=rnull, B=B, maxProcessor = 1)

## -----------------------------------------------------------------------------
dta=f(0.2) # Null hypothesis is false
twosample_test(dta, rnull=rnull, B=B, maxProcessor = 1)

## ----eval=FALSE---------------------------------------------------------------
# twosample_test_adjusted_pvalue(dta, rnull=rnull,
#                                B=B, maxProcessor = 1)

## ----eval=FALSE---------------------------------------------------------------
# twosample_power(f, c(0, 0.5), rnull=rnull, B=B, maxProcessor=1)

