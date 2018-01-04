\name{FastKDNeighbour}
\alias{FastKDNeighbour}
\title{K nearest neighbours using kd-trees}
\description{K nearest neighbours using kd-trees. Indices only.}
\usage{
FastKDNeighbour(query, ref, k)
}
\arguments{
  \item{query}{Matrix or data frame containing the set of query points
    where each row represents a point.}
  \item{ref}{Matrix or data frame containing the set of reference points
    where each row represents a point.}
  \item{k}{Number of nearest neighbours to search for.}
}
\value{
  Matrix containing the indices of the nearest neighbours in
  the reference set for each query set of points.
}
\examples{
## Find the nearest neighbour using a KD Tree
query <- matrix(rnorm(10), ncol = 2)
reference <- matrix(rnorm(10), ncol = 2)
FastKDNeighbour(query, reference, 3)
}
\author{Yee, Jeremy}

