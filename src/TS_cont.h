#ifndef TS_CONT_H
#define TS_CONT_H

#include <Rcpp.h>
Rcpp::NumericVector TS_cont(Rcpp::NumericMatrix x, 
                            Rcpp::NumericMatrix y,
                            Rcpp::Function knn,
                            int B=10000
                      );

#endif
