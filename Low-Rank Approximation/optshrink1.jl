using LinearAlgebra

function estimateDz(z, X)
    n, m = size(X)
    In = Matrix(I(n))
    Im = Matrix(I(m))

    z2IXXt = z^2*In - X*X'
    z2IXtX = z^2*Im - X'*X
    invz2XtX = inv(z2IXtX)
    invz2XXt = inv(z2IXXt)

    D1z = 1/n*tr(z*invz2XXt)
    D2z = 1/m*tr(z*invz2XtX)

    Dz = D1z*D2z

    return Dz
end

function estimateDtz(z, X)
    n, m = size(X)
    In = Matrix(I(n))
    Im = Matrix(I(m))

    z2IXXt = z^2*In - X*X';
    z2IXtX = z^2*Im - X'*X;
    invz2XtX = inv(z2IXtX);
    invz2XXt = inv(z2IXXt);

    D1z = 1/n*tr(z*invz2XXt);
    D2z = 1/m*tr(z*invz2XtX);


    D1zp = 1/n*tr(-2*z^2*invz2XXt^2+invz2XXt);
    D2zp = 1/m*tr(-2*z^2*invz2XtX^2+invz2XtX);

    Dpz = D1z*D2zp+D1zp*D2z;

    return Dpz
end

"""
Xh = optshrink1(Y::AbstractMatrix, r::Int)
Perform rank−r denoising of data matrix Y using the OptShrink method by Prof. Nadakuditi in this May 2014 IEEE Tr. on Info. Theory paper:
http://doi.org/10.1109/TIT.2014.2311661
In:
− 'Y' 2D array where Y=X+noise and goal is to estimate 'X'
− r estimated rank of 'X'
Out:
− Xh rank− r estimate of X using OptShrink weights for SVD components
This version works only if the size of  Y  is sufficiently small,
because it performs calculations involving arrays roughly of
size(Y'*Y) and size(Y*Y') , so neither dimension of Y can be large.

Citation:
Reference: Algorithm 1 in http://arxiv.org/abs/1306.6042
Author: Raj Rao Nadakuditi
"""
function optshrink1(Y::AbstractMatrix, r::Int)
    U_hat, S_hat, v_hat = svd(Y)

    n, m = size(Y)

    smallSigmaHats = S_hat
    bigSigmaRs = zeros(n - r, m - r)
    # Populate
    q = length(smallSigmaHats)
    index = 1
    for d in r+1:q
        bigSigmaRs[index, index] = smallSigmaHats[d]
        index = index + 1
    end
    # Diagonal(S_hat)[r+1:end,r+1:end]
    # @show size(bigSigmaRs)

    wOpts = zeros(r)
    for i in 1:r
        smallSigmaI = smallSigmaHats[i]
        top = estimateDz(smallSigmaI, bigSigmaRs)
        bottom = estimateDtz(smallSigmaI, bigSigmaRs)
        @show bottom
        wOpts[i] = -2 * (top / bottom)
    end

    Sopt = sum(wOpts[j] * U_hat[:,j] * (v_hat[:,j])' for j in 1:r)
    return Sopt
end
