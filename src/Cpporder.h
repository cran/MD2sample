#ifndef XYSORT_H
#define XYSort_H

#include <Rcpp.h>
std::vector<double> xysort(
   std::vector<double>& y, 
   std::vector<double>& x);

#endif

#ifndef ORDER_H
#define ORDER_H

#include <Rcpp.h>
std::vector<int> Order(
   std::vector<double>& x);

#endif

#ifndef RANK_H
#define RANK_H

#include <Rcpp.h>
std::vector<int> Rank(
   std::vector<double>& x);

#endif

