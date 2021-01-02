using LinearAlgebra
"""
lr_schatten(Y, reg::Real)
Compute the regularized low−rank matrix approximation as the minimizer over X
of '1/2 \\|Y − X\\|^2 + reg R(x)'
where 'R(X)' is the Schatten p−norm of X raised to the pth power, for p=1/2 ,
i.e., R(X) = \\sum_k (\\sigma_k(X))^{1/2}
In:
− 'Y' 'M by N' matrix
− 'reg' regularization parameter
Out:
− 'Xh' 'M by N' solution to above minimization problem
"""
function lr_schatten(Y, reg::Real)
    U, s, V = svd(Y)
    r = rank(Diagonal(s))
    Ur = U[:,1:r]; Vr = V[:,1:r]; sr = s[1:r]

    # Calculate each wk value
    wk = zeros(r)
    for k in 1:r
        ok = sr[k]
        if ok > (3/2)*(reg^(2/3))
            wk[k] = (4/3)*ok*((cos((1/3)*acos(-(3^(3/2)*reg)/(4*(ok^(3/2))))))^2)
        end
    end

    # Put weights into Er_hat
    Er_hat = Diagonal(wk)

    # Construct Xh
    Xh = Ur * Er_hat * Vr'

    return Xh
end
