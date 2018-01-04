## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Nearest Neighbours
################################################################################

Neighbour <- function(query, ref, k, build = "kdtree", cores = 0, checks = 1) {
    if (is.data.frame(query)) query <- as.matrix(query)
    if (is.data.frame(ref)) ref <- as.matrix(ref)
    .Call('_rflann_Neighbour', PACKAGE = 'rflann', query, ref, k,
          build, cores, checks)
}

