\name{RadiusSearch}
\alias{RadiusSearch}
\title{Radius searching}
\description{Radius searching}
\usage{
RadiusSearch(query, ref, radius, max_neighbour, build, cores, checks)
}
\arguments{
  \item{query}{Matrix containing the set of query points where each
    row represents a point.}
  \item{ref}{Matrix containing the set of reference points where each
    row represents a point.} 
  \item{radius}{Squared euclidean distance from each query point.}
  \item{max_neighbour}{Maximum number of points to look for within the
    radius of each query point.}
  \item{build}{String indicating the search structure to be used:
    "kdtree", "kmeans", "linear"}
  \item{cores}{Number of cpu cores to be used for searching.}
  \item{checks}{Number of checks during searching.}
}
\value{
  List containing:
  \item{indices}{List containing the index of points in the reference
    set that lie in the radius of each query point.}
  \item{distances}{List containing the corresponding squared distances.}
}
\examples{
## Radius searching
query <- matrix(rnorm(10), ncol = 2)
reference <- matrix(rnorm(10), ncol = 2)
RadiusSearch(query, reference, 1, 2, "kdtree", 0, 1)
}
\author{Yee, Jeremy}

