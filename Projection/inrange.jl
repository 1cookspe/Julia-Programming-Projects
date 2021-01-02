using LinearAlgebra;
"""
tf = inrange(B, z)
Return `true` or `false` depending on whether `z` is in the range of `B`
to within numerical precision. Must be as compute efficient as possible.
In:
* `B` a `M × N` matrix
* `z` vector of length `M`
"""
function inrange(B::AbstractMatrix, z::AbstractVector)
    # Compute SVD to get Ur
    (U,s,V) = svd(B)
    r = rank(Diagonal(s))
    Ur = U[:,1:r]

    # Calculate
    projection = Ur * (Ur' * z)

    # Check if projection is the same as z
    if isapprox(z, projection)
        # the same --> z in the range
        return true
    end
    return false
end
