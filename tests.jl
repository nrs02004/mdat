using Convex
using SCS
using ECOS

using Base.Test
include("setup.jl")

n = 100
x = sort(rand(n))

@test form_D(n) == PenaltyMatrix(x,1)

n = 3
T = 3
covariate_values = 1:n
tract_data = rand(n,T);

expected = ([1,2,3], [-1.0 1.0 0.0; 0.0 -1.0 1.0], [-1.0 1.0 0.0; 0.0 -1.0 1.0], [1.0, 1.0, 1.0], tract_data)
out = formProblem(covariate_values, tract_data, 1, 1)
for i=1:4
    @test_approx_eq out[i] expected[i]
end


##########

n = 3
T = 100
covariate_values = 1:n
tract_data = rand(n,T);

problem = formProblem(covariate_values, tract_data, 1, 1)
solveProblem(problem, 0.1, 0.1)

