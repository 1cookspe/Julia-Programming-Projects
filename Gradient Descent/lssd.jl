using LinearAlgebra
"""
x = lssd(A, b ; x0=zeros(size(A,2)), nIters::Int=10)
Perform steepest descent to solve the least squares problem
\\min_x \\| b − A x \\|_2
In:
*  A  a  m x n  matrix
*  b  a vector of length  m
Option:
*  x0  is the initial starting vector (of length  n ) to use; default  zeros
*  nIters  number of iterations to perform; default  10
Out:
*  x  a vector of length  n  containing the approximate solution
Notes:
Because this is a quadratic cost function, there is a closed−form solution
for the step size each iteration, so no "line search" procedure is needed.
A full−credit solution uses only *one* multiply by A and one by A' per iteration. """
function lssd(A, b ; x0=zeros(size(A,2)), nIters::Int=10)
    ak = [1.00] # default alpha value
    M = A' * A
    N = size(M)[2]
    P = I(N)
    xk = x0
    # initialize Ax to hold A*x_k+1
    Ax = A*xk

    for k in 1:nIters
        gk = A' * (Ax - b)
        dk = -P * gk
        Ad = A*dk
        ak = (-dk' * gk) / ((norm(Ad, 2))^2)[1,1]
        xk = xk + ak[1] * dk
        # Get next A * xk
        Ax += ak[1] * Ad
    end

    return xk
end
