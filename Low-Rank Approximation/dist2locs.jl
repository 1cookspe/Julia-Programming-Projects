using LinearAlgebra
"""
Xr = dist2locs(D, d)

In:
* `D` is an `n x n` matrix such that `D[i, j]` is the distance from object `i` to object `j`
* `d` is the desired embedding dimension.

Out:
* `Xr` is an `d x n` matrix whose columns contains the relative coordinates of the `n` objects

Note: MDS is only unique up to rotation and translation,
so we enforce the following conventions on Xr in this order:
* [ORDER] `Xr[i,:]` corresponds to ith largest eigenpair of `C' * C`
* [CENTER] The centroid of the coordinates is zero
* [SIGN] The largest magnitude element of `Xr[i,:]` is positive
"""
function dist2locs(D, d)
    J = size(D,1)

    # (1) First, find the matrix S, where s_ij = d_ij^2
    S = D.^2

    # (1.5) S should be symmetric, but in the case of noise, it might not be!
    S = 0.5 * (S + S') # Force S to be symmetric (optional, but helps with numerical stability)

    # (2) Next, de-mean
    # define the de-meaning operator, P_orth, to project onto the othogonal complement of Range(1_J)
    Jl = ones(J,1)
    P_orth = I(J) - (1/J) * Jl * Jl'
    G = (-1/2) * P_orth * S * P_orth # : Calculate G, the gram matrix, using P_orth

    # Calculate C_hat
    (_, s, V) = svd(G)
    Xr = Diagonal(sqrt.(s[1:d])) * V[:,1:d]'

    # (4) Get rid of the sign ambiguity
    # TODO: Force the largest magnitude element of each row of Xr to be positive
    rows = size(Xr)[1]
    cols = size(Xr)[2]
    for i in 1:rows
        maxRowVal = 0
        maxRowIndex = 0
        sign = 1
        for j in 1:cols
            if abs(Xr[i,j]) > maxRowVal
                maxRowVal = abs(Xr[i,j])
                maxRowIndex = j
            end
        end
        # Find sign
        if (Xr[i,maxRowIndex]) < 0
            sign = -1
        end
        # Multiply by sign
        for j in 1:cols
            Xr[i,j] = sign * Xr[i,j]
        end
    end

    return Xr
end
