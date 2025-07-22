#include <Rcpp.h>
using namespace Rcpp;
//' find pairwise Euclidean distances
//' 
//' @param dta A list of matrices
//' @keywords internal
//' @return A  matrix
// [[Rcpp::export]]
Rcpp::IntegerMatrix find_dist(Rcpp::List dta)   {
  NumericMatrix x=dta["x"];
  NumericMatrix y=dta["y"];
  int nx=x.nrow(), ny=y.nrow(), Dim=x.ncol(), n=nx+ny, i, j, k;
  double dst;
  IntegerMatrix Dz(n,n);
/* x with x */  
  for(i=0;i<(nx-1);++i) {
    for(j=i+1;j<nx;++j) {
      dst=0.0;
      for(k=0;k<Dim;++k) {
        dst=dst+(x(i,k)-x(j,k))*(x(i,k)-x(j,k));
      }
      Dz(i,j)=std::sqrt(dst)*10000;
      Dz(j,i)=Dz(i,j);
    }
  }
/* y with y */  
  for(i=0;i<(ny-1);++i) {
    for(j=i+1;j<ny;++j) {
      dst=0.0;
      for(k=0;k<Dim;++k) {
        dst=dst+(y(i,k)-y(j,k))*(y(i,k)-y(j,k));
      }
      Dz(i+nx,j+nx)=std::sqrt(dst)*10000;
      Dz(j+nx, i+nx)=Dz(i+nx,j+nx);
    }
  }

/* x with y */  
  for(i=0;i<nx;++i) {
    for(j=0;j<ny;++j) {
      dst=0.0;
      for(k=0;k<Dim;++k) {
        dst=dst+(x(i,k)-y(j,k))*(x(i,k)-y(j,k));
      }
      Dz(i,j+nx)=std::sqrt(dst)*10000;
      Dz(j+nx, i)=Dz(i, j+nx);
    }
  }
  return Dz;
} 
