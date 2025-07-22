#ifndef GEN_SIM_DATA_H
#define GEN_SIM_DATA_H

#include <Rcpp.h>
Rcpp::List gen_sim_data(Rcpp::List dta, Rcpp::List TSextra);

#endif

#ifndef GEN_CONT_H
#define GEN_CONT_H

#include <Rcpp.h>
Rcpp::List gen_cont(
              Rcpp::NumericMatrix x,
              Rcpp::NumericMatrix y, 
              Rcpp::List TSextra);
#endif

#ifndef GEN_DISC_H
#define GEN_DISC_H

#include <Rcpp.h>
Rcpp::List gen_disc(
              Rcpp::IntegerVector x, 
              Rcpp::IntegerVector y,
              Rcpp::NumericVector vals_x,
              Rcpp::NumericVector vals_y,
              Rcpp::List TSextra);

#endif
