## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Nearest Neighbours
################################################################################

Neighbour <- function(query, ref, k, build, cores, checks) {
    .Call('rflann_Neighbour', PACKAGE = 'rflann', query, ref, k, build, cores, checks)
}

