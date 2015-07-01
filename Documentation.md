## Multiple dimensional analysis of tracks

We are interested in solving the general family of problems that look like: 

min_{Theta} \frac{1}{2} || Y - \Theta ||_2 + \lambda_1 ||D_{1} ^ {(k1)} \Theta ||_n1 + \lambda_2 ||D_{2} ^{(k2)} \Theta ||_n2 

Where D_1, D_2 are matrices k-order difference matrices on the two dimensions of Theta (age, and track spacing), and D_1 enforces spatial smoothness and D_2 enforces the age structure we are interested in (e.g. piece-wise linear or piece-wise constant, etc.). n_1 and n_2 are the norms on these penalization.

## Specification

Prob = BuildProblem(k1, k2, dims)
SolveProblem(Prob, data)

Similar to the sklearn model/fit/predict API.

Prediction.

# Language? 

Julia great for our prototyping? Might we move to Python in the long run? 

Is there an implementation of ADMM in the Python cvxopt?
