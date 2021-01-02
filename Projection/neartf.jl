# sgablec
using LinearAlgebra
using Statistics: mean
"""
T = neartf(X)
Find nearest (in Frobenius norm sense) tight frame to matrix X.
In:
* 'X' : N × M nonzero matrix with N ≤ M
Out:
* 'T' N × M matrix that is the nearest tight frame to X
"""
function neartf(X::AbstractMatrix)
    # Perform SVD of X
    U, s, V = svd(X) # economy SVD
    # Form singular values matrix with the mean singular value
    r = size(U)[2]
    E = mean(s) * I(r)
    # Calculate T
    T = U * E * V'
    return T
end
