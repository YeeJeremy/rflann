\name{Neighbour}
\alias{Neighbour}
\title{K nearest neighbours}
\description{K nearest neighbours}
\usage{
Neighbour(query, ref, k, build = "kdtree", cores = 0, checks = 1)
}
\arguments{
  \item{query}{Matrix or data frame containing the set of query points
    where each row represents a point.}
  \item{ref}{Matrix or data frame containing the set of reference points
    where each row represents a point.}
  \item{k}{Number of nearest neighbours to search for.}
  \item{build}{String indicating the search structure to be used:
    "kdtree", "kmeans", "linear"}.
  \item{cores}{Number of cpu cores to be used for searching. If 0, then
    the maximum allowable cores are used.}
  \item{checks}{Number of checks during searching. Higher value gives
    better search precision but takes longer. See FLANN C++ manual for
    more details.}
}
\value{
  List containing:
  \item{indices}{Matrix containing the index of the nearest neighbours in
    the reference set for each query set of points}
  \item{distances}{Matrix containing the distances to the nearest neighbours}
}
\examples{
## Find the nearest neighbour using a KD Tree
query <- matrix(rnorm(10), ncol = 2)
reference <- matrix(rnorm(10), ncol = 2)
Neighbour(query, reference, 3, "kdtree", 0, 1)
}
\author{Yee, Jeremy}

