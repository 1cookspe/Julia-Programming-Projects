"""
out = shrink_p_1_2(v, reg::Real)
Compute minimizer of 1/2 |v − x|^2 + reg |x|^p
for p=1/2 when v is real and nonnegative.
In:
*  v  scalar, vector, or array of (real, nonnegative) input values
*  reg  regularization parameter
Out:
*  xh   solution to minimization problem for each element of  v
(same size as  v )
"""
function shrink_p_1_2(v, reg::Real)
    M = size(v, 1)
    N = size(v, 2)

    xh = zeros(M, N)

    # Loop through each element of xh (if more than one)
    for i in 1:M
        for j in 1:N
            vi = v[i,j]
            if vi > (3/2)*(reg^(2/3))
                xh[i,j] = (4/3)*vi*(cos((1/3)*acos(((-3^(3/2))*reg)/(4*(vi^(3/2))))))^2
            end
        end
    end

    return xh
end
