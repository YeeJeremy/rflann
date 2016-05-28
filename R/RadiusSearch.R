## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Radius Searching
################################################################################

RadiusSearch <- function(query, ref, radius, max_neighbour, build, cores, checks) {
    .Call('rflann_RadiusSearch', PACKAGE = 'rflann', query, ref, radius,
          max_neighbour, build, cores, checks)
}
