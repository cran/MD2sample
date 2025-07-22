#include <Rcpp.h>
using namespace Rcpp;
//' Find test statistics for discrete data
//' @param x a vector of integers.
//' @param y a vector of integers.
//' @param vx a vector of real.
//' @param vy a vector of real.
//' @param TSextra list with extra infos
//' @keywords internal
//' @return A numeric vector with test statistics
// [[Rcpp::export]]
Rcpp::NumericVector TS_disc(Rcpp::IntegerVector x,
                            Rcpp::IntegerVector y,
                            Rcpp::NumericVector vx,
                            Rcpp::NumericVector vy,
                            Rcpp::List TSextra) 
  {  
  Rcpp::CharacterVector methods=Rcpp::CharacterVector::create("KS", "K", "CvM", "AD", "NN", "AZ", "BF");
  int const nummethods=methods.size();
  int i, j, k, m=x.size(), nx, ny, sx, sy, n;
  Function organize=TSextra["organize"];
  NumericMatrix dta(m,4);
  dta(_,0)=vx;
  dta(_,1)=vy;
  dta(_,2)=x;
  dta(_,3)=y;
  dta=organize(dta);
  NumericVector TS(nummethods);

  vx=sort_unique(vx);
  nx=vx.size();
  sx=sum(x);
  vy=sort_unique(vy);
  ny=vy.size();
  sy=sum(y);
  n=sx+sy;
  NumericMatrix O_x(ny, nx), O_y(ny, nx), ecdf_x(ny, nx), 
                ecdf_y(ny, nx), ecdf_z(ny, nx);
  IntegerMatrix counts_x(ny, nx), counts_y(ny, nx); 
  k=-1;
  for(j=0;j<nx;++j) {
    for(i=0;i<ny;++i) {     
         ++k;
         O_x(ny-i-1, j) = dta(k, 2);
         O_y(ny-i-1, j) = dta(k, 3);
    }  
  } 

  counts_x(ny-1,0)=O_x(ny-1,0); 
  counts_y(ny-1,0)=O_y(ny-1,0);
  for(i=ny-2;i>=0;--i) {
    counts_x(i,0)=counts_x(i+1,0)+O_x(i,0);
    counts_y(i,0)=counts_y(i+1,0)+O_y(i,0);
  }  
  for(i=1;i<nx;++i) {
    counts_x(ny-1,i)=counts_x(ny-1,i-1)+O_x(ny-1, i);
    counts_y(ny-1,i)=counts_y(ny-1,i-1)+O_y(ny-1, i);
  }
  for(i=ny-2;i>=0;--i) {   
     for(j=1;j<nx;++j) {
       counts_x(i,j)=counts_x(i,j-1)+counts_x(i+1,j)-counts_x(i+1,j-1)+O_x(i,j);
       counts_y(i,j)=counts_y(i,j-1)+counts_y(i+1,j)-counts_y(i+1,j-1)+O_y(i,j);     
     }
  }  
  for(i=0;i<ny;++i) {
    for(j=0;j<nx;++j) {
      ecdf_x(i,j)=counts_x(i,j)/double(sx);   
      ecdf_y(i,j)=counts_y(i,j)/double(sy);
      ecdf_z(i,j)=(counts_x(i,j)+counts_y(i,j))/double(n);
    }
  }
/*  Kolmogorov-Smirnov and Kuiper*/  

  NumericVector tmp(2);
  for(i=0;i<ny;++i) {
    for(j=0;j<nx;++j) {
       double a=ecdf_x(i,j)-ecdf_y(i,j);
       if(a<tmp(0)) tmp(0)=a;
       if(a>tmp(1)) tmp(1)=a;
    }
  }
  TS(1)=std::abs(tmp(0))+std::abs(tmp(1));
  if(std::abs(tmp(0))<std::abs(tmp(1)))
       TS(0)=std::abs(tmp(1));
  else  TS(0)=std::abs(tmp(0));

/* Cramer-vonMises and Anderson-Darling */  
  
  TS(2)=0.0;
  TS(3)=0.0;
  for(i=0;i<ny;++i) {
    for(j=0;j<nx;++j) {
       tmp(0)=ecdf_x(i,j)-ecdf_y(i,j);
       TS(2)=TS(2)+(O_x(i,j)+O_y(i,j))*tmp(0)*tmp(0);
       if(ecdf_z(i,j)>0 && ecdf_z(i,j)<1)
          TS(3)=TS(3)+(O_x(i,j)+O_y(i,j))*tmp(0)*tmp(0)/ecdf_z(i,j)/(1-ecdf_z(i,j));
    }
  }     
  TS(2)=TS(2)*sx*sy/n/n;
  TS(3)=TS(3)*sx*sy/n/n;

/* Nearest Neighbors */

  NumericMatrix Exy(ny,nx);
  for(i=0;i<ny;++i) 
    for(j=0;j<nx;++j) 
       Exy(i,j)=(O_x(i,j)+O_y(i,j))/double(n);
  TS(4)=0;
  for(i=0;i<ny;++i) {
    for(j=0;j<nx;++j) {
       if(i>0) {
          TS(4)=TS(4)+O_x(i, j)*std::abs(O_x(i-1,j)-sx*Exy(i-1,j))
                     +O_y(i, j)*std::abs(O_y(i-1,j)-sy*Exy(i-1,j));
       }
       if(i<ny-1) {
          TS(4)=TS(4)+O_x(i, j)*std::abs(O_x(i+1,j)-sx*Exy(i+1,j))
                     +O_y(i, j)*std::abs(O_y(i+1,j)-sy*Exy(i+1,j));
       }
       if(j>0) {
          TS(4)=TS(4)+O_x(i, j)*std::abs(O_x(i,j-1)-sx*Exy(i,j-1))
                     +O_y(i, j)*std::abs(O_y(i,j-1)-sy*Exy(i,j-1));
       }
       if(j<nx-1) {
          TS(4)=TS(4)+O_x(i, j)*std::abs(O_x(i,j+1)-sx*Exy(i,j+1))
                     +O_y(i, j)*std::abs(O_y(i,j+1)-sy*Exy(i,j+1));
       }
    }
  }
  TS(4)=TS(4)/4.0/(sx+sy);
 
/*  Aslan-Zech, Baringhaus-Franz*/   

  NumericMatrix taz(2,3);
  int a1,a2,a3;
  double ds;
  n=dta.nrow();
  for(i=0;i<(n-1);++i) {
    for(j=(i+1);j<n;++j) {
      ds=(dta(i,0)-dta(j,0))*(dta(i,0)-dta(j,0))+
                (dta(i,1)-dta(j,1))*(dta(i,1)-dta(j,1));
      a1=dta(i,2)*dta(j,2), a2=dta(i,3)*dta(j,3), a3=dta(i,2)*dta(j,3);
      if(ds>0) taz(0,0)=taz(0,0)-a1*log(ds);         
      taz(1,0)=taz(1,0)+a1*sqrt(ds);         
      if(ds>0) taz(0,1)=taz(0,1)-a2*log(ds);         
      taz(1,1)=taz(1,1)+a2*sqrt(ds);         
      if(ds>0) taz(0,2)=taz(0,2)-a3*log(ds);         
      taz(1,2)=taz(1,2)+a3*sqrt(ds);             
    } 
  }
  TS(5)=std::abs(taz(0,0)/sx/sx-taz(0,2)/sx/sy)+
        std::abs(taz(0,1)/sy/sy-taz(0,2)/sx/sy);
  TS(6)=std::abs(taz(1,0)/sx/sx-taz(1,2)/sx/sy)+
        std::abs(taz(1,1)/sy/sy-taz(1,2)/sx/sy); 
  TS.names()=methods;
  return TS;
} 
