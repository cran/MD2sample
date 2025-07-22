#include <Rcpp.h>
#include "calcTS.h"
#include "gen_sim_data.h"
#include "transform01.h"
using namespace Rcpp;

//' Find the power of various continuous tests via simutation or permutation.
//' 
//' @param rxy a function that generates x and y data.
//' @param xparam  arguments for r1.
//' @param yparam  arguments for r2.
//' @param TS routine to calculate test statistics for non-chi-square tests
//' @param typeTS indicator for type of test statistics
//' @param TSextra additional info passed to TS, if necessary
//' @param B =1000 number of simulation runs
//' @keywords internal
//' @return A list values of test statistics
// [[Rcpp::export(rng=false)]]
List powerC(Function rxy, 
            NumericVector xparam,                       
            NumericVector yparam,
            Function TS,
            int typeTS,
            List TSextra,
           int B=1000) { 
/* Find out how many tests are to be done, sample sizes etc. */  
  
  int j,l, m, nl=xparam.size();
  NumericVector TS_data;
  Function dist=TSextra["dist"];
  List dta;
  if(typeTS<=3) dta=rxy(xparam(0),yparam(0));
  else {
    NumericMatrix tmpdata=rxy(xparam(0),yparam(0));
    dta=List::create(Named("vals_x")=tmpdata(_,0),
                     Named("vals_y")=tmpdata(_,1),
                     Named("x")=tmpdata(_,2),
                     Named("y")=tmpdata(_,3));
  }
  if(typeTS==1) TSextra["distances"]=dist(dta);
  TS_data = calcTS(dta, TS, typeTS, TSextra);
  int const nummethods=TS_data.size();
  int cn=-1;
  NumericMatrix realdta(B*nl, nummethods), simdta(B*nl, nummethods), paramalt(B*nl,2);
  NumericVector TS_sim(nummethods);
/*l loop over values in xparam */  
  for(l=0;l<nl;++l) {
/*m loop over simulation runs */  
    for(m=0;m<B;++m) {
       ++cn;
/*  create new data set  */            
       if(typeTS<=3) dta=rxy(xparam(l),yparam(l));
       else {
          NumericMatrix tmpdata=rxy(xparam(l),yparam(l));
          dta=List::create(Named("vals_x")=tmpdata(_,0),
                          Named("vals_y")=tmpdata(_,1),
                          Named("x")=tmpdata(_,2),
                          Named("y")=tmpdata(_,3));
      } 
      if(typeTS<=3) {
         if(TSextra["DoTransform"]) {
           if(TSextra["ParametricBootstrap"]) TSextra["rawdta"]=dta;
           dta=transform01(dta);
         }   
         TSextra["distances"]=dist(dta);
       }
      TS_data = calcTS(dta, TS, typeTS, TSextra);
      paramalt(cn,0)=xparam(l);
      paramalt(cn,1)=yparam(l);
      for(j=0;j<nummethods;++j) realdta(cn,j)=TS_data(j);
      GetRNGstate();
      dta=gen_sim_data(dta, TSextra);
      PutRNGstate();
      if(typeTS==1) TSextra["distances"]=dist(dta);
      TS_sim = calcTS(dta, TS, typeTS, TSextra);
      for(j=0;j<nummethods;++j) simdta(cn,j)=TS_sim(j);
    } 
  }
  return List::create(Named("Data")=realdta, 
               Named("Simulated")=simdta,
               Named("paramalt")=paramalt);
}

