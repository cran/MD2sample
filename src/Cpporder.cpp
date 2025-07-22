#include <Rcpp.h>
using namespace Rcpp;

//' sort vector y by values in vector x
//' 
//' @param y numeric vector
//' @param x numeric vector
//' @keywords internal
//' @return numeric vector
// [[Rcpp::export]]  
std::vector<double> xysort(std::vector<double>& y, std::vector<double>& x) {
    int n=x.size();
    std::vector<int> Index(n);
    std::iota(Index.begin(), Index.end(), 0);
    std::sort(Index.begin(), Index.end(),
              [&](int A, int B) -> bool {
                return x[A] < x[B];
              });
    std::vector<double> out(n);
    for(int i=0;i<n;++i) out[i]=y[Index[i]];
    return out;
  }

//' equivalent to R command order
//' 
//' @param x numeric vector
//' @keywords internal
//' @return integer vector
// [[Rcpp::export]]  
std::vector<int> Order(std::vector<double>& x) {
   int n=x.size();
   std::vector<int> Index(n), y(n);
   for(int i=0;i<n;++i) y[i]=i;
   std::iota(Index.begin(), Index.end(), 0);
   std::sort(Index.begin(), Index.end(),
             [&](int A, int B) -> bool {
               return x[A] < x[B];
             });
   std::vector<int> out(n);
   for(int i=0;i<n;++i) out[i]=y[Index[i]];
   return out;
 }
//' equivalent to R command rank
//' 
//' @param x numeric vector
//' @keywords internal
//' @return integer vector
// [[Rcpp::export]]
std::vector<int> Rank(std::vector<double>& x) {
  int n=x.size();
  std::vector<int> Index(n), y(n);
/* First ordering  */
  for(int i=0;i<n;++i) y[i]=i;
  std::iota(Index.begin(), Index.end(), 0);
  std::sort(Index.begin(), Index.end(),
            [&](int A, int B) -> bool {
              return x[A] < x[B];
            });
  std::vector<int> out(n);
  for(int i=0;i<n;++i) out[i]=y[Index[i]]+1;
/* Second ordering  */ 
  std::vector<int> Index1(n);
  for(int i=0;i<n;++i) y[i]=i;
  std::iota(Index1.begin(), Index1.end(), 0);
  std::sort(Index1.begin(), Index1.end(),
            [&](int A, int B) -> bool {
              return out[A] < out[B];
            });
  for(int i=0;i<n;++i) out[i]=y[Index1[i]]+1;
  
  return out;
}
