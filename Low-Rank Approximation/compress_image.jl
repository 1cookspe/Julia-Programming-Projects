using LinearAlgebra
"""
Ac, r = compress_image(A, p)
In:
*A m×nmatrix
* p scalar in (0, 1]
Out:
* Ac a m × n matrix containing a compressed version of A that can be represented using at most (100 * p)% as many bits
required to represent A
* r the rank of Ac
"""
function compress_image(A, p)
    m = size(A)[1]
    n = size(A)[2]

    k = Int(floor((p * n * m) / (m + n + 1)))

    U, s, V = svd(A)
    r = rank(Diagonal(s))

    bound = min(k, r)

    # Loop and fill optimal A
    Ac = zeros(m, n)
    for i in 1:bound
        Ac = Ac + s[i] * U[:,i] * (V[:,i])'
    end

    rc = rank(Ac)

    return Ac, rc
end
