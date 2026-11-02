#include <Rcpp.h>
#include "calcTS.h"
#include "gen_sim_data.h"
using namespace Rcpp;

//' run tests using either simulation or permutation.
//' 
//' @param dta a list with the data
//' @param TS routine to calculate test statistics for non-chi-square tests 
//' @param typeTS type of a test statistic
//' @param TSextra additional info passed to TS, if necessary
//' @param B =5000, number of simulation runs.
//' @keywords internal
//' @return A list with test statistics and p values
// [[Rcpp::export(rng=false)]]
List testC(List dta, 
          Function TS,
          int typeTS,
          List TSextra,
          int B=5000) { 
  NumericVector TS_data;
  TS_data = calcTS(dta, TS, typeTS, TSextra);
  int const nummethods=TS_data.size();
  int i, j;
  NumericVector pvals(nummethods), TS_sim(nummethods), stats(nummethods);
  Function dist=TSextra["dist"];
  CharacterVector TSmethods=TS_data.names();
  stats.names() = TSmethods; 
  pvals.names() = TSmethods;
  for(i=0;i<nummethods;++i) {
    stats(i)=TS_data(i);  
    pvals(i)=0.0;  
  }
/*  if B=0 return just the test statistics */    
  if(B==0) return List::create(Named("statistics")=stats);
/* find test statistic for simulated or simuted data  */ 
  for(j=0;j<B;++j) {
    GetRNGstate();
    List simdta=gen_sim_data(dta, TSextra);
    PutRNGstate();
    if( (typeTS==1) || (typeTS==3) ) TSextra["distances"]=dist(simdta);
    TS_sim= calcTS(simdta, TS, typeTS, TSextra);
/*  count simulated statistics at least as large as the observed one. 
    A small relative tolerance makes sure that ties are counted even if
    the two values differ by round-off error  */
    for(i=0;i<nummethods;++i) {
       double tol=1e-10*std::max(1.0, std::abs(TS_data(i)));
       if(TS_sim(i)>=TS_data(i)-tol) pvals(i)=pvals(i)+1.0;
    }
  }

/* proportion of simulated statistics >= observed. The +1 correction
   p=(1+B*pvals)/(B+1) is applied in twosample_test, after the results
   of all processors have been combined. */  
  for(i=0;i<nummethods;++i) pvals(i)=pvals(i)/B;
  return  List::create(Named("statistics")=stats, Named("p.values")=pvals);
}

