"""
x = lsgd(A, b ; mu=0, x0=zeros(size(A,2)), nIters::Int=200)
Performs gradient descent to solve the least squares problem: \\argmin_x 0.5 \\| b − A x \\|_2
In:
− A m×n matrix
− b vector of length m
Option:
− mu step size to use, and must satisfy 0 < mu < 2 / \\sigma_1(A)^2 to guarantee convergence,
where \\sigma_1(A) is the first (largest) singular value. Ch.5 will explain a default value for mu
− x0 is the initial starting vector (of length n ) to use. Its default value is all zeros for simplicity.
− nIters is the number of iterations to perform (default 200)
Out:
− x vector of length n containing the approximate LS solution
"""
function lsgd(A, b ; mu::Real=0, x0=zeros(size(A,2)), nIters::Int=200)
    # Create x vector
    n = size(A)[2] # Get number of columns of A
    x = x0

    # Loop through each iteration to find x(k+1)
    for k in range(1, stop = nIters)
        x = x - (mu * A' * (A * x - b))
    end

    return x
end
