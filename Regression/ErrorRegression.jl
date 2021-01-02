using Plots; default(markerstrokecolor=:auto)
using Random: seed!
using LinearAlgebra;

# get T
T = LinRange(0, 2, 16)
t = collect(T)

resNorm = zeros(1,2)
fitErr = zeros(1,2)

# Calculate f
f = zeros(length(t),1)
n = size(t)[1]
for i = 1:n
    f[i] = 0.5 * exp(0.8 * t[i])
end

# Generate noise
seed!(3); e = randn(length(T))
y = f + e

# Scatter plot
scatter(t,y,label="Data", ylim=[-1,4])
plot!(xlabel = "t")
plot!(ylabel = "y = f + e", legend = :bottomright)

# Polynomial of degree 15
deg = 15 # polynomial degree
# Construct A
A = zeros(n, deg+1)
for i = 0:deg
    A[:,i+1] = t.^i
end
# Calculate coefficients
x_hat = A \ y
# Calculate function
tSamp = collect((0:1000)/500)
function calculateFunction()
    accum = zeros(length(tSamp),1)
    for i = 0:deg
        accum += x_hat[i+1] * tSamp.^i
    end
    return accum
end
p15 = calculateFunction()
plot!(tSamp,p15,label="p15(t)")
resNorm[1] = norm(A*x_hat - y)
fitErr[1] = norm(A*x_hat - f)

# Polynomial of degree 2
deg = 2 # polynomial degree
# Construct A
A = zeros(n, deg+1)
for i = 0:deg
    A[:,i+1] = t.^i
end
# Calculate coefficients
x_hat = A \ y
# Calculate function
tSamp = collect((0:1000)/500)
function calculateFunction()
    accum = zeros(length(tSamp),1)
    for i = 0:deg
        accum += x_hat[i+1] * tSamp.^i
    end
    return accum
end
p2 = calculateFunction()
plot!(tSamp,p2,label="p2(t)")
resNorm[2] = norm(A*x_hat - y)
fitErr[2] = norm(A*x_hat - f)

# Save plot
savefig("ErrorRegression.png")
