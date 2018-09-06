// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Radius search
////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <RcppArmadillo.h>
#include <string>
#include "flann/flann.hpp"

//[[Rcpp::export]]
Rcpp::List RadiusSearch(Rcpp::NumericMatrix query,
                        Rcpp::NumericMatrix ref,
                        double radius,
                        int max_neighbour,
                        std::string build,
                        int cores,
                        int checks) {
  const std::size_t n_dim = query.ncol();
  const std::size_t n_query = query.nrow();
  const std::size_t n_ref = ref.nrow();
  // Column major to row major
  arma::mat qquery(n_dim, n_query);
  {
    arma::mat temp_q(query.begin(), n_query, n_dim, false);
    qquery = arma::trans(temp_q);
  }
  flann::Matrix<double> q_flann(qquery.memptr(), n_query, n_dim);
  arma::mat rref(n_dim, n_ref);
  {
    arma::mat temp_r(ref.begin(), n_ref, n_dim, false);
    rref = arma::trans(temp_r);
  }
  flann::Matrix<double> ref_flann(rref.memptr(), n_ref, n_dim);
  // Setting the flann index params
  flann::IndexParams params;
  if (build == "kdtree") {
    params = flann::KDTreeSingleIndexParams(1);
  } else if (build == "kmeans") {
    params = flann::KMeansIndexParams(2, 10, flann::FLANN_CENTERS_RANDOM, 0.2);
  } else if (build == "linear") {
    params = flann::LinearIndexParams();
  }
  // Perform the radius search
  flann::Index<flann::L2<double> > index(ref_flann, params);
  index.buildIndex();
  std::vector< std::vector<int> >
      indices_flann(n_query, std::vector<int>(max_neighbour));
  std::vector< std::vector<double> >
      dists_flann(n_query, std::vector<double>(max_neighbour));
  flann::SearchParams search_params;
  search_params.cores = cores;
  search_params.checks = checks;
  search_params.max_neighbors = max_neighbour;
  index.radiusSearch(q_flann, indices_flann, dists_flann, radius,
                     search_params);
  return Rcpp::List::create(Rcpp::Named("indices") = indices_flann,
                            Rcpp::Named("distances") = dists_flann);
}
