#include <Rcpp.h>
using namespace Rcpp;
//' Finds the empirical distribution function
//' @param dta a matrix of data points
//' @param pts a matrix of evaluation points
//' @keywords internal
//' @return a numeric vector
// [[Rcpp::export]]
NumericVector mdecdf(NumericMatrix dta, NumericMatrix pts) {
   int i, j, k, d=dta.ncol(), n=dta.nrow(), m=pts.nrow();
   double tmp=1.0/n;
   NumericVector out(m);
   for(j=0;j<m;++j) {
     for(i=0;i<n;i++) {
       k=0;
       while(k<d && dta(i, k)<=pts(j, k)) ++k;
       if(k==d) out[j]=out[j]+tmp;   
     }      
   }     
   return out;
 }
