using LinearAlgebra;

"""
R = orthcompnull(A, X)
Project each column of X onto the orthogonal complement of the null space of the input matrix A .
In:
* A M×N matrix
* X vector of length N , or matrix with N rows and many columns
Out:
*  R  : vector or matrix of size ??? (you determine this)
For full credit, your solution should be computationally efficient!
"""
function orthcompnull(A, X)
    # Orthogonal component of null space = basis of Vr
    # Compute SVD to get Vr
    U, s, V = svd(A, full=true)
    r = rank(Diagonal(s))
    Vr = V[:,1:r]
    # Projection is Vr * Vr'
    Pr = Vr * Vr'
    # Project onto null space
    R = Pr * X
end
