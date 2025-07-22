#include <Rcpp.h>
using namespace Rcpp;
#include "mdecdf.h"
//' Find test statistics for continuous data
//' 
//' @param x A matrix.
//' @param y A matrix
//' @param TSextra list with some helper functions
//' @keywords internal
//' @return A numeric vector with test statistics
// [[Rcpp::export]]
Rcpp::NumericVector TS_cont(
        Rcpp::NumericMatrix x, 
        Rcpp::NumericMatrix y,
        Rcpp::List TSextra) 
  {
  Rcpp::CharacterVector methods=Rcpp::CharacterVector::create("KS", "K", "CvM", "AD", "NN1", "NN5", "AZ", "BF", "BG");
  int const nummethods=methods.size();
  int nx=x.nrow(), ny=y.nrow(), n=nx+ny, i, j, Dim=x.ncol();
  double tmp1, tmp2;
  NumericVector TS(nummethods);
  TS.names() =  methods;

/*  Setup */  
  NumericMatrix z(n, Dim);
  for(i=0;i<nx;++i)
     for(j=0;j<Dim;++j)
       z(i, j) = x(i, j);
  for(i=0;i<ny;++i)
     for(j=0;j<Dim;++j)
       z(i+nx, j) = y(i, j);

/*  Kolmogorov-Smirnov and Kuiper*/

  NumericVector edf_x(n), edf_y(n), edf_z(n);
  edf_x=mdecdf(x,z);
  edf_y=mdecdf(y,z);
  edf_z=(edf_x*nx+edf_y*ny)/n;
  NumericVector edfxy = edf_x-edf_y; 
  tmp1=0.0;
  tmp2=0.0;
  for(i=0;i<n;++i) {
     if(edfxy(i)<tmp1) tmp1=edfxy(i);
     if(edfxy(i)>tmp2) tmp2=edfxy(i);
  }
  tmp1=std::abs(tmp1);
  tmp2=std::abs(tmp2);
  TS(0)=tmp1;
  if(tmp2>tmp1) TS(0)=tmp2;
  TS(1)=tmp1+tmp2;

/* Cramer-vonMises and Anderson-Darling */    
  
  TS(2)=0.0;
  TS(3)=0.0;
  for(i=0;i<n;++i) {
      tmp1=edf_x(i)-edf_y(i);  
      TS(2)=TS(2)+tmp1*tmp1;
      if(edf_z(i)>0 && edf_z(i)<1)
         TS(3)=TS(3)+tmp1*tmp1/edf_z(i)/(1-edf_z(i));
  }
  TS(2)=TS(2)*nx*ny/n/n;
  TS(3)=TS(3)*nx*ny/n/n;

/*  Nearest Neighbors, Schilling, Narsky*/ 
  Function knn=TSextra["knn"];
  IntegerMatrix NN = knn(z);
  tmp1=0.0;
  tmp2=0.0;
  for(i=0;i<nx;++i) {
    if(NN(i,0)<=nx) {
      tmp1=tmp1+1;
      tmp2=tmp2+1;
    }
    for(j=1;j<Dim;++j) 
      if(NN(i,j)<=nx) tmp2=tmp2+1;
  }
  TS(4)=tmp1/double(nx);
  TS(5)=tmp2/double(nx);
  tmp1=0.0;
  tmp2=0.0;  
  for(i=0;i<ny;++i) {
    if(NN(i+nx,0)>nx){
        tmp1=tmp1+1;
        tmp2=tmp2+1;
    }
    for(j=1;j<Dim;++j) {
        if(NN(i+nx,j)>nx) tmp2=tmp2+1;
    }    
  }
  TS(4)=TS(4)+tmp1/double(ny);
  TS(5)=TS(5)+tmp2/double(ny);
  
/*  Aslan-Zech, Baringhaus-Franz, Biswas-Ghosh */  

  NumericVector tmp(2), BG(3);
  NumericMatrix Dz=TSextra["distances"];
  for(i=0;i<(nx-1);++i) {
     for(j=i+1;j<nx;++j) {
       if(Dz(i,j)>0) tmp(0)=tmp(0)-log(Dz(i,j));
       tmp(1)=tmp(1)+std::sqrt(Dz(i,j));
     }   
  }
  TS(6)=tmp(0)/nx/nx;
  TS(7)=tmp(1)/nx/nx;
  BG(0)=2*tmp(1)/nx/(nx-1);
  tmp(0)=0.0;
  tmp(1)=0.0;
  for(i=0;i<(ny-1);++i) {
    for(j=i+1;j<ny;++j) {
      if(Dz(i,j)>0) tmp(0)=tmp(0)-log(Dz(nx+i,nx+j));
      tmp(1)=tmp(1)+std::sqrt(Dz(nx+i,nx+j));
    }
  }
  TS(6)=TS(6)+tmp(0)/ny/ny;
  TS(7)=TS(7)+tmp(1)/ny/ny;
  BG(1)=2*tmp(1)/ny/(ny-1);
  tmp(0)=0.0;
  tmp(1)=0.0;
  for(i=0;i<nx;++i) {
    for(j=0;j<ny;++j) {
      if(Dz(i,j)>0) tmp(0)=tmp(0)-log(Dz(i,nx+j));
      tmp(1)=tmp(1)+std::sqrt(Dz(i,nx+j));
    }
  }
  TS(6)=TS(6)-tmp(0)/nx/ny;
  TS(7)=TS(7)-tmp(1)/nx/ny;
  TS(7)=TS(7)*nx*ny/n;
  TS(7)=-TS(7)/100.0;
  BG(2)=tmp(1)/nx/ny;
  TS(8)=(BG(0)-BG(2))*(BG(0)-BG(2))+(BG(1)-BG(2))*(BG(1)-BG(2))/100.0;
  return TS;
} 
