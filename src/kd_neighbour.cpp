// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Fast nearest neighbour using single kd-tree
////////////////////////////////////////////////////////////////////////////////

//[[Rcpp::interfaces(r, cpp)]]

#include <RcppArmadillo.h>
#include "flann/flann.hpp"

//[[Rcpp::export]]
arma::imat FastKDNeighbour(Rcpp::NumericMatrix query,
                           Rcpp::NumericMatrix ref,
                           const std::size_t& k) {
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
  // Finding the nearest neighbours using single KD tree
  flann::Index<flann::L2<double> > index(ref_flann, flann::KDTreeSingleIndexParams(1));
  index.buildIndex();
  flann::Matrix<int> indices_flann(new int[n_query * k], n_query, k);
  flann::Matrix<double> dists_flann(new double[n_query * k], n_query, k);
  flann::SearchParams search_params;
  search_params.cores = 0;
  search_params.checks = 1;
  index.knnSearch(q_flann, indices_flann, dists_flann, k, search_params);
  arma::imat indices(indices_flann.ptr(), k, n_query, true);
  delete[] indices_flann.ptr();
  delete[] dists_flann.ptr();
  return indices.t() + 1;
}

