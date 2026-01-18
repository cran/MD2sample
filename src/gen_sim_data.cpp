#include <Rcpp.h>
#include "gen_sim_data.h"
#include "transform01.h"
using namespace Rcpp; 

//' simulate new data
//' @param dta data set
//' @param TSextra extra stuff
//' @keywords internal
//' @return A list of permuted vectors
// [[Rcpp::export]]
Rcpp::List gen_sim_data(List dta, List TSextra) {
   List newdta; 
   if(dta.size()==2) newdta=gen_cont(dta["x"], dta["y"], TSextra);
   if(dta.size()==4) {
     newdta=gen_disc(dta["x"], dta["y"], dta["vals_x"], dta["vals_y"], TSextra);
   }
   return newdta;
} 

// [[Rcpp::export]]
int getI(NumericVector p) {
  double tmp=R::runif(0,1);
  int j=0;
  while (tmp>p(j)) ++j;
  return j;
}
                               
//' simulate continuous data
//' @param x first data set
//' @param y second data set
//' @param TSextra extra stuff
//' @keywords internal
//' @return A list of permuted vectors
// [[Rcpp::export]]
Rcpp::List gen_cont(NumericMatrix x, NumericMatrix y, List TSextra) {
  int nx, ny, n, i, j;
  nx=x.nrow();
  ny=y.nrow();
  int Dim=x.ncol();
  n=nx+ny;
  CharacterVector A= CharacterVector::create("rnull");
  CharacterVector B=TSextra.names();
  if(in(A, B)[0]) { 
    Function rnull=TSextra["rnull"];
    List rawdta=TSextra["rawdta"];
    List simdta=rnull(rawdta);
    if(TSextra["DoTransform"]) simdta=transform01(simdta);
    return simdta;
  }
  
  NumericMatrix permx(nx, Dim), permy(ny, Dim), z(n, Dim);
  IntegerVector Index(n);
  List permdta=List::create(Named("x")=x,Named("y")=y);
  for(i=0;i<nx;++i) {
    Index(i)=i;
    for(j=0;j<Dim;++j) z(i,j)=x(i,j);
  }  
  for(i=0;i<ny;++i) {
    Index(i+nx)=i+nx;
    for(j=0;j<Dim;++j) z(i+nx,j)=y(i,j);
  } 
  Index = Rcpp::sample(Index, n);
  for(int i=0;i<nx;++i) {
    for(j=0;j<Dim;++j) permx(i,j)=z(Index[i],j);
  }
  for(i=0;i<ny;++i) {
    for(j=0;j<Dim;++j) permy(i,j)=z(Index[i+nx],j);
  }    
  permdta["x"]=permx;
  permdta["y"]=permy;
  return permdta;
}

//' simulate new discrete data
//' @param x first data set, counts
//' @param y second data set, counts
//' @param vals_x values of discrete random variable x
//' @param vals_y values of discrete random variable y
//' @param TSextra extra stuff
//' @keywords internal
//' @return A list of permuted vectors
// [[Rcpp::export]]
Rcpp::List gen_disc(IntegerVector x, 
                    IntegerVector y,
                    NumericVector vals_x,
                    NumericVector vals_y,
                    List TSextra) {
  int i, j, k=x.size();
  List out;
  Function organize=TSextra["organize"];
  CharacterVector A= CharacterVector::create("rnull");
  CharacterVector B=TSextra.names();
  if(in(A, B)[0]) {
    NumericMatrix simdta(k,4);
    colnames(simdta) = CharacterVector::create("vals_x", "vals_y", "x", "y");
    simdta(_,0)=vals_x;
    simdta(_,1)=vals_y;  
    simdta(_,2)=x;
    simdta(_,3)=y;
    Function rnull=TSextra["rnull"];
    simdta=rnull(simdta);
    return List::create(Named("vals_x")=simdta(_,0),
                        Named("vals_y")=simdta(_,1),
                        Named("x")=simdta(_,2),
                        Named("y")=simdta(_,3));
  }
  int samplingmethod=TSextra["samplingmethod"];
  int nx, ny; 
  nx=0;
  ny=0;
  for(i=0;i<k;++i) {
    nx=nx+x(i); 
    ny=ny+y(i);
  }
  if(samplingmethod==1) {
    NumericVector vx1=sort_unique(vals_x), vy1=sort_unique(vals_y);
    IntegerVector vx(k),vy(k);
    int mx=0,my=0;
    for(i=0;i<vx1.size();++i)
      for(j=0;j<vy1.size();++j) {
        vx[mx]=j;    
        vy[mx]=i;  
        ++mx;
    }
    NumericMatrix xL(nx,1), yL(ny,1);
    mx=0,my=0;
    int Mx=max(vx)+1;
    for(i=0;i<k;++i) {
         for(j=0;j<x(i);++j) {
         xL(mx,0)=vx(i)+Mx*vy(i);
         ++mx;
       }
       for(j=0;j<y(i);++j) {
         yL(my,0)=vx(i)+Mx*vy(i);
         ++my;
       }
    }
    GetRNGstate();
    out=gen_cont(xL, yL, TSextra);
    PutRNGstate();
    NumericMatrix sx=out["x"], sy=out["y"];
    IntegerVector simx(k), simy(k);
    for(i=0;i<nx;++i) {
      mx=sx(i,0);
      simx(mx)=simx(mx)+1;
    }  
    for(i=0;i<ny;++i) {
      my=sy(i,0);
      simy(my)=simy(my)+1;
    }
    out=List::create(Named("x")=simx,
                          Named("y")=simy,
                          Named("vals_x")=vals_x,
                          Named("vals_y")=vals_y);
  } 
  if(samplingmethod==2) {
    out=List::create(Named("x")=x,
                          Named("y")=y,
                          Named("vals_x")=vals_x,
                          Named("vals_y")=vals_y);
    if(samplingmethod==2) {
      double p=double(nx)/double(nx+ny);
      NumericVector pr(x.size());
      pr(0)=double(x(0)+y(0));
      for(i=1;i<k;++i) pr(i)=pr(i-1)+double(x(i)+y(i));
      for(i=0;i<k;++i) pr(i)=pr(i)/pr(k-1);
      IntegerVector tt(1),newx(k);
      for(i=0;i<k;++i) {
        tt=Rcpp::rbinom(1, x(i)+y(i), p);
        newx(i)=tt(0);
      }   
      int d=sum(x)-sum(newx);
      while (d<0) {
        j=getI(pr);
        newx(j)=newx(j)-1;
        d=d+1;
      }
      while (d>0) {
        j=getI(pr);
        newx(j)=newx(j)+1;
        d=d-1;
      }
      out=List::create(Named("x")=newx,
                       Named("y")=x+y-newx,
                       Named("vals_x")=vals_x,
                       Named("vals_y")=vals_y);
    }
  }
  return out;
}

