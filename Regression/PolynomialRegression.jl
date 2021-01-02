using CSV
using DataFrames
using Plots; default(markerstrokecolor=:auto)
using LinearAlgebra;
using Random: seed!

T = LinRange(0, 2, 16)
t = Array{Float64,1}(collect(T))

resNorm = zeros(1,2)

# Construct f
f = zeros(length(t),1)
b = f
n = size(t)[1]
for i = 1:n
    f[i] = 0.5 * exp(0.8 * t[i])
end
# Scatter plot
scatter(t,f,label="Data", ylim=[-1,4])
plot!(xlabel = "t_i")
plot!(ylabel = "b_i", legend = :bottomright)

# Polynomial of degree 15
deg = 15 # polynomial degree
# Construct A
A = zeros(n, deg+1)
for i = 0:deg
    A[:,i+1] = t.^i
end
# Calculate coefficients
x_hat15 = A \ b
# Calculate function
tSamp = collect((0:1000)/500)
function calculateFunction()
    accum = zeros(length(tSamp),1)
    for i = 0:deg
        accum += x_hat15[i+1] * tSamp.^i
    end
    return accum
end
p15 = calculateFunction()
plot!(tSamp,p15,label="p15(t)")
resNorm[1] = norm(A*x_hat15 - b)
#
# Polynomial of degree 2
deg = 2 # polynomial degree
# Construct A
A = zeros(n, deg+1)
for i = 0:deg
    A[:,i+1] = t.^i
end
# Calculate coefficients
x_hat2 = A \ b
# Calculate function
tSamp = collect((0:1000)/500)
function calculateFunction()
    accum = zeros(length(tSamp),1)
    for i = 0:deg
        accum += x_hat2[i+1] * tSamp.^i
    end
    return accum
end
p2 = calculateFunction()
plot!(tSamp,p2,label="p2(t)")
# Norm calculation
resNorm[2] = norm(A*x_hat2 - b)

#
# Save plot
savefig("P15P2.png")
