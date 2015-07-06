## Forms a difference matrix, for use with fused lasso/trend filtering

function form_D(n)
    D = zeros(n-1,n)
    for i=1:(n-1)
        D[i,i] = -1
        D[i,i+1] = 1
    end
    return D
end


## Forms an "inverse difference matrix", for use with unequally spaced trend filtering

function DxInv(x, order)
    D = zeros(length(x)-order,length(x)-order)
    for i=1:(length(x)-order)
        D[i,i] = 1/(x[i+order] - x[i])
    end
    return D
end

## x has to be sorted from smallest to largest
function PenaltyMatrix(x, degree)
    n = length(x)
    Pen_mat = form_D(n)
    counter = 1
    while(counter < degree)
        Pen_mat = form_D(n-counter) * DxInv(x,counter) * Pen_mat
        counter = counter + 1
    end
    return Pen_mat
end


## Assume tract_values are ordered
function formProblem(covariate_values, tract_data, location_degree, covariate_degree, tract_locations=None)

    if(tract_locations == None)
        tract_locations = 1:(size(tract_data)[2])
    end

    (combined_tract_data, weights, combined_covs)  = combine(covariate_values, tract_data)
    perm = sortperm(combined_covs)
    location_smoother = PenaltyMatrix(tract_locations, location_degree)
    covariate_smoother = PenaltyMatrix(combined_covs[perm], covariate_degree)

    return (perm, location_smoother, covariate_smoother, weights, combined_tract_data)
end

function solveProblem(problem, location_lambda, covariate_lambda)
    (perm, location_smoother, covariate_smoother, weights, tract_data) = problem

    (n, p) = size(tract_data)
    fitted_values = Variable(n,p)
    problem = minimize(quad_form(fitted_values - tract_data, diagm(vec(weights)))
                       + covariate_lambda * norm(covariate_smoother * fitted_values[perm,:],1)
                       + location_lambda * norm(location_smoother * (fitted_values'),1))
    out = solve!(problem, ECOSSolver(max_it=100))
    return fitted_values.value
end


## Checks each row of a matrix for NAs --- returns row-numbers with NAs

function hasNaN(X)
    (n,p) = size(X)
    remove = zeros(n)
    for i=1:n
        remove[i] = sum(isnan(X[i,:])) >= 1
    end
return remove
end

function fitter(X, D, D0, lam1, lam2)
    (n, p) = size(X)
    b = Variable(p,n)
    problem = minimize(sum_squares(b - X') + lam1*norm(D*b,1) + lam2*norm(D0*(b'),1))
    out = solve!(problem, SCSSolver(max_iters = 10000, normalize = 0))
    return b.value
end

## Fits a 2-d smoother. D determines smoothness along columns, D0 determines smoothness wrt a covariate (age in our case)
## Also takes in a weight vector (w), used because we have averaged observations with the same age

function fitter_group(X, w, D, D0, lam1, lam2)
    (n, p) = size(X)
    b = Variable(p,n)
    problem = minimize(quad_form(b' - X, diagm(vec(w))) + lam1*norm(D*b,1) + lam2*norm(D0*(b'),1))
    out = solve!(problem, SCSSolver(max_iters = 10000, normalize = 0))
    return b.value
end

## Fits a 1-d smoother + linear model in age. D determines smoothness along columns
## Also takes in a weight vector (w), used because we have averaged observations with the same age


function fitter_linear(X, age, w, D, lam1, lam2)
    (n, p) = size(X)
    b = Variable(1,p)
    beta = Variable(1,p)
    problem = minimize(quad_form(ones(n,1) * b + age * beta - X, diagm(vec(w))) + lam1*norm(D*b',1) + lam2*norm((beta),1))
    out = solve!(problem, SCSSolver(max_iters = 100000, normalize = 0))
    return {"b"=>b.value, "beta"=>beta.value}
end


## Averages observations with the same age, and returns a new X vector, and weights corresponding to number of terms averaged

function combine(covariate, X)
    (n,p) = size(X)
    unique_covariate = unique(covariate)
    weights = Float32[]
    new_X = Array(Float32,0,p)
    for cov in unique_covariate
        push!(weights, sum(cov.==covariate))
        new_X = vcat(new_X, mean(X[find(covariate.==cov),:],1))
    end
    return (new_X, weights, unique_covariate)
end

