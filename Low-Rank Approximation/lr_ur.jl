# sgablec
"""
Xh = lr_ur(Y, Ur, K::Int)
Compute the rank-at-most-K best approximation of Y = U_r Σ_r V_r'
In:
- Y : M × N matrix
- Ur : M × r matrix with first r left singular vectors of Y
- K maximum rank of Xh (any nonnegative integer)
Out:
- Xh : M × N best approximation, having rank <= K
"""
function lr_ur(Y::AbstractMatrix, Ur::AbstractMatrix, K::Int)
    # Get rank from the number of columns of Ur
    M, N = size(Y)
    R = N

    # Check default cases
    if K >= R # just return the original matrix
        return Y
    elseif K == 0 # return zero matrix
        return zeros(M, N)
    end

    # Loop K times to construct Xk
    Xk = zeros(ComplexF64, M, N)
    for k in 1:K
        sigmaV = zeros(ComplexF64, 1, R)
        for r in 1:N
            sigmaV[r] = Ur[:,k]'*Y[:,r]
        end
        # Multiply sigmaV with the r-th column of Ur
        Xk += Ur[:,k] * sigmaV
    end

    return Xk
end
