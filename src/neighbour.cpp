// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Nearest Neighbours
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <string>
#include "flann/flann.hpp"

Rcpp::List Neighbour(Rcpp::NumericMatrix query_,
                     Rcpp::NumericMatrix ref_,
                     std::size_t k,
                     std::string build,
                     std::size_t cores,
                     std::size_t checks) {
  std::size_t n_dim = query_.ncol();
  std::size_t n_query = query_.nrow();
  std::size_t n_ref = ref_.nrow();
  // Column major to row major
  arma::mat query(n_dim, n_query);
  {
    arma::mat temp_q(query_.begin(), n_query, n_dim, false);
    query = arma::trans(temp_q);
  }
  flann::Matrix<double> q_flann(query.memptr(), n_query, n_dim);
  arma::mat ref(n_dim, n_ref);
  {
    arma::mat temp_r(ref_.begin(), n_ref, n_dim, false);
    ref = arma::trans(temp_r);
  }
  flann::Matrix<double> ref_flann(ref.memptr(), n_ref, n_dim);
  // Setting the flann index params
  flann::IndexParams params;
  if (build == "kdtree") {
    params = flann::KDTreeSingleIndexParams(1);
  } else if (build == "kmeans") {
    params = flann::KMeansIndexParams(2, 10, flann::FLANN_CENTERS_RANDOM, 0.2);
  } else if (build == "linear") {
    params = flann::LinearIndexParams();
  }
  // Finding the nearest neighbours
  flann::Index<flann::L2<double> > index(ref_flann, params);
  index.buildIndex();
  flann::Matrix<int> indices_flann(new int[n_query * k], n_query, k);
  flann::Matrix<double> dists_flann(new double[n_query * k], n_query, k);
  flann::SearchParams search_params;
  search_params.cores = cores;
  search_params.checks = checks;
  index.knnSearch(q_flann, indices_flann, dists_flann, k, search_params);
  arma::imat indices(indices_flann.ptr(), k, n_query, true);
  arma::mat dists(dists_flann.ptr(), k, n_query, true);
  delete[] indices_flann.ptr();
  delete[] dists_flann.ptr();
  return Rcpp::List::create(Rcpp::Named("indices") = indices.t() + 1,
                            Rcpp::Named("distances") = dists.t());
}

// Export to R
RcppExport SEXP rflann_Neighbour(SEXP query_SEXP,
                                 SEXP ref_SEXP,
                                 SEXP kSEXP,
                                 SEXP buildSEXP,
                                 SEXP coresSEXP,
                                 SEXP checksSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type
        query_(query_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type
        ref_(ref_SEXP);
    Rcpp::traits::input_parameter< std::size_t >::type
        k(kSEXP);
    Rcpp::traits::input_parameter< std::string >::type
        build(buildSEXP);
    Rcpp::traits::input_parameter< std::size_t >::type
        cores(coresSEXP);
    Rcpp::traits::input_parameter< std::size_t >::type
        checks(checksSEXP);
    __result = Rcpp::wrap(Neighbour(query_, ref_, k, build, cores, checks));
    return __result;
END_RCPP
}
