"""
x = lsngd(A, b ; x0 = zeros(size(A,2)), nIters = 200, mu = 0)
Perform Nesterov−accelerated gradient descent to solve the LS problem \\argmin_x 0.5 \\| A x − b \\|_2
In:
− A m×n matrix
− b vector of length m
Option:
− x0 initial starting vector (of length n ) to use; default 0 vector. − nIters number of iterations to perform; default 200.
− mu step size, must satisfy 0 < \\mu \\leq 1 / \\sigma_1(A)^2
to guarantee convergence, where \\sigma_1(A) is the first (largest) singular value.
Ch.5 will explain a default value for mu .
Out:
 x  vector of length  n  containing the approximate solution
"""
function lsngd(A::AbstractMatrix{<:Number}, b::AbstractVector{<:Number} ; x0::AbstractVector{<:Number} = zeros(eltype(b), size(A,2)),
nIters::Int = 200, mu::Real = 0)
    t = 1
    x = x0
    z = x0

    for k in range(1, stop = nIters)
        tk = t # t(k)
        t = (1 + sqrt(1 + 4 * t^2)) / 2 # t(k+1)
        xk = x # x(k)
        x = z - mu * A' * (A * z - b) # x(k+1)
        z = x + ((tk - 1) / t) * (x - xk)
    end

    return x
end
