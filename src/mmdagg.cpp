#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace Rcpp;

/*
 MMD test aggregated over a collection of Gaussian-kernel bandwidths.

 For each bandwidth sigma_k the unbiased estimator MMD_u^2 (Gretton et al. 2012)
 is computed. Because the size of MMD_u^2 depends strongly on the bandwidth,
 each value is divided by an estimate of its standard deviation under H0,

   Var_0(MMD_u^2) ~ 2 s_k^2 [ 1/(n(n-1)) + 1/(m(m-1)) + 2/(nm) ],

 where s_k^2 is the mean of the squared, doubly centered kernel values
 over all pairs of the pooled sample. s_k^2 depends only on the pooled
 sample, so it is invariant under permutations. The test statistic is the
 maximum of the standardized values. As in MMDAgg (Schrab et al. 2023) the
 bandwidths are sigma_med * 2^l, l in [-4, 4], with sigma_med the median
 inter-point distance of the pooled sample; the p value is found by
 permutation of this maximum.
*/

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
  int const n_bandwidths = 10;
  double const log2_min = -4.0;
  double const log2_max = 4.0;
  if (y.ncol() != d)
    stop("x and y must have the same number of columns.");
  if (n < 2 || m < 2)
    stop("x and y must each have at least 2 rows.");

  int N = n + m;
  /* pooled sample and squared distances */
  NumericMatrix Z(N, d);
  for (int i = 0; i < n; i++)
    for (int k = 0; k < d; k++) Z(i, k) = x(i, k);
  for (int i = 0; i < m; i++)
    for (int k = 0; k < d; k++) Z(n + i, k) = y(i, k);

  std::vector<double> D2((size_t) N * N, 0.0);
  std::vector<double> dists;
  dists.reserve((size_t) N * (N - 1) / 2);
  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      double s = 0.0;
      for (int k = 0; k < d; k++) {
        double diff = Z(i, k) - Z(j, k);
        s += diff * diff;
      }
      D2[(size_t) i * N + j] = s;
      D2[(size_t) j * N + i] = s;
      dists.push_back(std::sqrt(s));
    }
  }
  size_t mid = dists.size() / 2;
  std::nth_element(dists.begin(), dists.begin() + mid, dists.end());
  double sigma_med = dists[mid];
  if (!(sigma_med > 0.0)) sigma_med = 1.0;

  double const varfac = 2.0 * (1.0 / (double(n) * (n - 1)) +
                               1.0 / (double(m) * (m - 1)) +
                               2.0 / (double(n) * m));
  std::vector<double> K((size_t) N * N), rowmean(N);
  double max_stat = R_NegInf;

  for (int b = 0; b < n_bandwidths; b++) {
    double frac = double(b) / double(n_bandwidths - 1);
    double sigma = sigma_med * std::pow(2.0, log2_min + frac * (log2_max - log2_min));
    double c = 1.0 / (2.0 * sigma * sigma);
    /* kernel matrix and its row means */
    double grand = 0.0;
    for (int i = 0; i < N; i++) {
      double rs = 0.0;
      for (int j = 0; j < N; j++) {
        double kv = (i == j) ? 1.0 : std::exp(-D2[(size_t) i * N + j] * c);
        K[(size_t) i * N + j] = kv;
        rs += kv;
      }
      rowmean[i] = rs / N;
      grand += rs;
    }
    grand /= (double(N) * N);
    /* unbiased MMD^2 */
    double Kxx = 0.0, Kyy = 0.0, Kxy = 0.0;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        if (i != j) Kxx += K[(size_t) i * N + j];
    for (int i = n; i < N; i++)
      for (int j = n; j < N; j++)
        if (i != j) Kyy += K[(size_t) i * N + j];
    for (int i = 0; i < n; i++)
      for (int j = n; j < N; j++)
        Kxy += K[(size_t) i * N + j];
    double mmd2 = Kxx / (double(n) * (n - 1)) + Kyy / (double(m) * (m - 1)) -
                  2.0 * Kxy / (double(n) * m);
    /* null variance from doubly centered kernel */
    double s2 = 0.0;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        if (i == j) continue;
        double kc = K[(size_t) i * N + j] - rowmean[i] - rowmean[j] + grand;
        s2 += kc * kc;
      }
    s2 /= (double(N) * (N - 1));
    if (!(s2 > 0.0)) continue;
    double stat = mmd2 / std::sqrt(varfac * s2);
    if (stat > max_stat) max_stat = stat;
  }
  if (!R_finite(max_stat)) max_stat = 0.0;
  return max_stat;
}
