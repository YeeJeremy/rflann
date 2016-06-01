## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Radius Searching
################################################################################

RadiusSearch <- function(query, ref, radius, max_neighbour, build = "kdtree",
                         cores = 0, checks = 1) {
    if (is.data.frame(query)) query <- as.matrix(query)
    if (is.data.frame(ref)) ref <- as.matrix(ref)    
    output <- .Call('rflann_RadiusSearch', PACKAGE = 'rflann', query,
                    ref, radius, max_neighbour, build, cores, checks)
    output$indices <- lapply(output$indices, function(x){x+1})
    return(output)
}
