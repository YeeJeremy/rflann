// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Radius search
////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <RcppArmadillo.h>
#include <string>
#include "flann/flann.hpp"

Rcpp::List RadiusSearch(Rcpp::NumericMatrix query_,
                        Rcpp::NumericMatrix ref_,
                        double radius,
                        int max_neighbour,
                        std::string build,
                        int cores,
                        int checks) {
  const std::size_t n_dim = query_.ncol();
  const std::size_t n_query = query_.nrow();
  const std::size_t n_ref = ref_.nrow();
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

// Export to R
RcppExport SEXP rflann_RadiusSearch(SEXP query_SEXP,
                                    SEXP ref_SEXP,
                                    SEXP radiusSEXP,
                                    SEXP max_neighbourSEXP,
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
    Rcpp::traits::input_parameter< double >::type
        radius(radiusSEXP);
    Rcpp::traits::input_parameter< int >::type
        max_neighbour(max_neighbourSEXP);
    Rcpp::traits::input_parameter< std::string >::type
        build(buildSEXP);
    Rcpp::traits::input_parameter< int >::type
        cores(coresSEXP);
    Rcpp::traits::input_parameter< int >::type
        checks(checksSEXP);
    __result = Rcpp::wrap(RadiusSearch(query_, ref_, radius,
                                       max_neighbour, build, cores, checks));
    return __result;
END_RCPP
}
