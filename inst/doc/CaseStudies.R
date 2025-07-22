## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(MD2sample)
cases=MD2sample::case.studies(ReturnCaseNames=TRUE)

## -----------------------------------------------------------------------------
multiple.graphs = function (px, py,xname=" ") {
  egg::ggarrange(px, py, ncol = 2, nrow = 1)
} 

## -----------------------------------------------------------------------------
for(i in seq_along(cases)) {
   if(!endsWith(cases[i], "D2")) next
   tmp=case.studies(i, 2000)
   print(cases[i])
   p=tmp$param_alt[2]
   if(startsWith(cases[i],"Dalitz")) p=10*p
   dta=tmp$f(p)
   dtax=data.frame(x=dta$x[,1], y=dta$x[,2])
   dtay=data.frame(x=dta$y[,1], y=dta$y[,2])
   px=ggplot2::ggplot(dtax, ggplot2::aes(x=x, y=y))+
     ggplot2::geom_point(color="blue", alpha=0.5, size=0.5)+
     ggplot2::labs(x=expression(x[1]), y=expression(x[2])) 
     
   py=ggplot2::ggplot(dtay, ggplot2::aes(x=x, y=y))+
     ggplot2::geom_point(color="red", alpha=0.5)+
     ggplot2::labs(x=expression(x[1]), y=expression(x[2])) 
  multiple.graphs(px, py, cases[i])
}

## -----------------------------------------------------------------------------
for(i in seq_along(cases)) {
   if(!endsWith(cases[i], "M")) next
   tmp=case.studies(i, 2000)
   print(cases[i])
   p=tmp$param_alt[2]
   if(startsWith(cases[i],"Dalitz")) p=10*p
   dta=tmp$f(p)
   dtax=data.frame(x=dta$x[,1], y=dta$x[,2])
   dtay=data.frame(x=dta$y[,1], y=dta$y[,2])
   px=ggplot2::ggplot(dtax, ggplot2::aes(x=x, y=y))+
     ggplot2::geom_point(color="blue", alpha=0.5, size=0.5)+
     ggplot2::labs(x=expression(x[1]), y=expression(x[2])) 
     
   py=ggplot2::ggplot(dtay, ggplot2::aes(x=x, y=y))+
     ggplot2::geom_point(color="red", alpha=0.5)+
     ggplot2::labs(x=expression(x[1]), y=expression(x[2])) 
  multiple.graphs(px, py, cases[i])
}

