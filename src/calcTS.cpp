#include <Rcpp.h>
using namespace Rcpp;
//' This function calculates the test statistics
//' @param  dta data set
//' @param  TS routine
//' @param  typeTS format of TS
//' @param  TSextra list passed to TS function
//' @keywords internal
//' @return A vector of numbers
// [[Rcpp::export]]
NumericVector calcTS(
     Rcpp::List dta, 
     Rcpp::Function TS,
     int typeTS,
     Rcpp::List TSextra) {
   NumericVector TS_data;
   if( (typeTS==1) || (typeTS==3) ) TS_data=TS(dta["x"], dta["y"], TSextra); 
   if(typeTS==2) TS_data=TS(dta["x"], dta["y"]);   
   if((typeTS==4) || (typeTS==6)) TS_data=TS(dta["x"], dta["y"], dta["vals_x"], dta["vals_y"], TSextra);  
   if(typeTS==5) TS_data=TS(dta["x"], dta["y"],dta["vals_x"], dta["vals_y"]); 
   return  TS_data;
 }
 
 
