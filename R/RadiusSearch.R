## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Radius Searching
################################################################################

RadiusSearch <- function(query, ref, radius, max_neighbour, build, cores, checks) {
    output <- .Call('rflann_RadiusSearch', PACKAGE = 'rflann', query, ref,
                    radius, max_neighbour, build, cores, checks)
    output$indices <- lapply(output$indices, function(x){x+1})
    return(output)
}
