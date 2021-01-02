using LinearAlgebra
"""
T = nearptf(X)
Find nearest (in Frobenius norm sense) Parseval tight frame to matrix X . In:
X : N×M matrix
Out:
T N × M matrix that is the nearest Parseval tight frame to X """
function nearptf(X::AbstractMatrix)
    # Get components of SVD
    U, s, V = svd(X)
    r = rank(Diagonal(s))
    Ur = U[:,1:r]
    Vr = V[:,1:r]

    # We know that the singular values of the Parseval Frame are all equal to 1
    # Construct sigma matrix as all 1's
    sigmaR = Diagonal(ones(r))
    # Calculate Parseval frame using the same U and V from X
    T = Ur * sigmaR * Vr'

    return T
end
