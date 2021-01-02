using LinearAlgebra

function calcD(z, S, K, L)
    # Compute D(z, X)

    term1 = 1 / (K * L)
    term2 = 0
    term3 = 0
    for k in 1:L
        sk = S[k]
        term2 += z / (z^2 - sk^2)
        term3 += z / (z^2 - sk^2)
    end
    term2 += (K - L) / z

    D = term1 * term2 * term3
    return D
end

function calcDt(z, S, K, L)
    # Compute D'(z, X)

    term1 = 1 / (K * L)
    term2 = 0
    term3 = 0
    term4 = 0
    term5 = term1
    term7 = 0
    for k in 1:L
        sk = S[k]
        term2 += z / (z^2 - sk^2)
        term3 += (z^2) / ((z^2 - sk^2)^2)
        term4 += 1 / (z^2 - sk^2)
        term7 += (z^2) / ((z^2 - sk^2)^2)
    end
    term6 = term2
    term2 += (K - L) / z
    # term3 += K - L
    term3 *= -2
    term7 += (K - L) / z^2
    term7 *= -2
    term8 = term4
    term8 += ((K - L) / z^2)

    Dt = term1 * term2 * (term3 + term4) + term5 * term6 * (term7 + term8)
    return Dt
end

"""
Xh = optshrink2(Y::AbstractMatrix, r::Int)
Perform rank− r denoising of data matrix Y using the OptShrink method
by Prof. Nadakuditi in this May 2014 IEEE Tr. on Info. Theory paper: http://doi.org/10.1109/TIT.2014.2311661
In:
− 'Y' 2D array where Y=X+noise and goal is to estimate X − r estimated rank of X
Out:
− Xh rank− r estimate of X using OptShrink weights for SVD components
This version works even if one of the dimensions of Y is large, as long as the other is sufficiently small.
"""
function optshrink2(Y::AbstractMatrix, r::Int)
    U, s, V = svd(Y)

    n, m = size(Y)

    # Get dimensions
    K = m - r
    L = n - r
    if (n >= m) # tall
        K = n - r
        L = m - r
    end

    # Calculate wOpts in loop
    wOpts = zeros(r)
    for i in 1:r
        # Calculate D
        D = calcD(s[i], s[r+1:end], K, L)
        Dt = calcDt(s[i], s[r+1:end], K, L)
        wOpts[i] = -2 * (D / Dt)
    end

    Sopt = sum(wOpts[j] * U[:,j] * (V[:,j])' for j in 1:r)
    return Sopt
end
