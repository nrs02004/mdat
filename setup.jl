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

function combine(ages, X)
    (n,p) = size(X)
    unique_ages = unique(ages)
    weights = Float32[]
    new_X = Array(Float32,0,p)
    for age in unique_ages
        push!(weights, sum(age.==ages))
        new_X = vcat(new_X, mean(X[find(ages.==age),:],1))
    end
    return {"X"=>new_X, "w"=>weights, "age"=>unique_ages}
end

