#include <Rcpp.h>
using namespace Rcpp;
//' transform data to interval (0,1).
//' 
//' @param dta a list.
//' @keywords internal
//' @return a list 
// [[Rcpp::export]]
Rcpp::List transform01(List dta) {
   NumericMatrix tmpx = as<NumericMatrix>(dta["x"]);
   NumericMatrix tmpy = as<NumericMatrix>(dta["y"]);
   int k, Dim=tmpx.ncol();
   NumericMatrix x(tmpx.nrow(), Dim), y(tmpy.nrow(), Dim);
   for(int j=0;j<Dim;++j) {
     for(int i=0;i<x.nrow();++i) x(i,j)=tmpx(i,j);
     for(int i=0;i<y.nrow();++i) y(i,j)=tmpy(i,j);
   }
   double tmp;
   NumericVector minmax(2);
   for(k=0;k<Dim;++k) {
     minmax(0)=min(x(_,k));
     tmp=min(y(_,k));
     if(tmp<minmax(0)) minmax(0)=tmp;
     minmax(1)=max(x(_,k));
     tmp=max(y(_,k));
     if(tmp>minmax(1)) minmax(1)=tmp;
     x(_,k)=(x(_,k)-minmax(0))/(minmax(1)-minmax(0));
     y(_,k)=(y(_,k)-minmax(0))/(minmax(1)-minmax(0));
   }
   return Rcpp::List::create(Named("x")=x, Named("y")=y);
 }


