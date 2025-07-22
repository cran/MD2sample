#ifndef TS_DISC_H
#define TS_DISC_H

#include <Rcpp.h>
Rcpp::NumericVector TS_disc(Rcpp::IntegerVector x,
                            Rcpp::IntegerVector y,
                            Rcpp::NumericVector vx,
                            Rcpp::NumericVector vy,
                            Rcpp::List TSextra
                      );

#endif
