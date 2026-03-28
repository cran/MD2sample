
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace Rcpp;

inline double sq_dist_row(const NumericMatrix& A, int i,
                          const NumericMatrix& B, int j) {
  int d = A.ncol();
  double s = 0.0;
  for (int k = 0; k < d; k++) {
    double diff = A(i, k) - B(j, k);
    s += diff * diff;
  }
  return s;
}

inline double gauss_kernel(double sqd, double sigma) {
  return std::exp(-sqd / (2.0 * sigma * sigma));
}

double pooled_median_sigma(const NumericMatrix& X, const NumericMatrix& Y) {
  int n = X.nrow();
  int m = Y.nrow();
  int d = X.ncol();

  NumericMatrix Z(n + m, d);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < d; j++)
      Z(i, j) = X(i, j);

  for (int i = 0; i < m; i++)
    for (int j = 0; j < d; j++)
      Z(n + i, j) = Y(i, j);

  int N = Z.nrow();
  std::vector<double> dists;
  dists.reserve((size_t) N * (N - 1) / 2);

  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      dists.push_back(std::sqrt(sq_dist_row(Z, i, Z, j)));
    }
  }

  if (dists.empty()) return 1.0;

  size_t mid = dists.size() / 2;
  std::nth_element(dists.begin(), dists.begin() + mid, dists.end());
  double med = dists[mid];

  return (med > 0.0 ? med : 1.0);
}

double compute_mmd2(const NumericMatrix& X,
                    const NumericMatrix& Y,
                    double sigma) {
  int n = X.nrow();
  int m = Y.nrow();

  double Kxx = 0.0, Kyy = 0.0, Kxy = 0.0;

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      Kxx += gauss_kernel(sq_dist_row(X, i, X, j), sigma);
  Kxx /= static_cast<double>(n) * static_cast<double>(n);

  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++)
      Kyy += gauss_kernel(sq_dist_row(Y, i, Y, j), sigma);
  Kyy /= static_cast<double>(m) * static_cast<double>(m);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      Kxy += gauss_kernel(sq_dist_row(X, i, Y, j), sigma);
  Kxy /= static_cast<double>(n) * static_cast<double>(m);

  return Kxx + Kyy - 2.0 * Kxy;
}

double agg_statistic(const NumericMatrix& X,
                     const NumericMatrix& Y,
                     const std::vector<double>& sigmas) {
  double max_stat = R_NegInf;
  for (size_t k = 0; k < sigmas.size(); k++) {
    double s = compute_mmd2(X, Y, sigmas[k]);
    if (s > max_stat) max_stat = s;
  }
  return max_stat;
}

//' Find test statistic for MMD test
//' 
//' @param x A matrix 
//' @param y A matrix 
//' @keywords internal
//' @return A numeric vector with test statistics
// [[Rcpp::export]]
double mmdagg(NumericMatrix x,
                     NumericMatrix y) {
  int n = x.nrow();
  int m = y.nrow();
  int d = x.ncol();
  int n_bandwidths = 10;
  double log2_min = -4.0;
  double log2_max = 4.0;
  if (y.ncol() != d)
    stop("x and y must have the same number of columns.");
  if (n < 2 || m < 2)
    stop("x and y must each have at least 2 rows.");
  if (n_bandwidths < 1)
    stop("n_bandwidths must be at least 1.");

  double sigma_med = pooled_median_sigma(x, y);

  std::vector<double> sigmas(n_bandwidths);
  if (n_bandwidths == 1) {
    sigmas[0] = sigma_med;
  } else {
    for (int i = 0; i < n_bandwidths; i++) {
      double frac = static_cast<double>(i) / static_cast<double>(n_bandwidths - 1);
      double exponent = log2_min + frac * (log2_max - log2_min);
      sigmas[i] = sigma_med * std::pow(2.0, exponent);
    }
  }

  return agg_statistic(x, y, sigmas);
 
}
