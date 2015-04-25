using Gadfly
using Color
using Cairo
#using DataFrames
using Convex
using SCS
using ECOS
include("setup.jl")

tract_data = readdlm("data/All_Tracts_FA.csv",',')
age_data = readdlm("data/Subject_Age.csv",',')

ord = sortperm(age_data[:,1])
age_data = age_data[ord,1]

for tract_num = 1:28

    X = tract_data[ord,(2:99) + 100*(tract_num-1)]

    keep = find(hasNaN(X).!=1)
    X = X[keep,:]
    
    age = age_data[keep,1];

    junk = combine(age, X); age = junk["age"]; w = junk["w"]; X = junk["X"];
    
    (n, p) = size(X)
    lam1 = 0.4; lam2s = 0:10:80
    
    D_a1 = form_D(n); D_age = DxInv(age,1); D_a2 = form_D(n-1);
    D0 = D_a2 * D_age * D_a1
    D1 = form_D(p); D2 = form_D(p-1); D3 = form_D(p-2);
    D = D3 * D2 * D1;
    
    fits = [fitter_linear(X, age, w, D, lam1, lam2) for lam2 in lam2s]
    
    #splatting

    println("$tract_num")
    
    colors = colormap("Purples", int(floor(1.5*n)));

    for i=1:length(fits)
        foo = ones(n,1) * fits[i]["b"] + age * fits[i]["beta"]
        pl = plot([layer(x=1:p,y=foo[j,:], Geom.line, Theme(default_color=colors[j+floor(0.5*n)])) for j in 1:n]...)
        draw(PDF("linPlots/tract$(tract_num)/plot$(i).pdf", 12cm, 8cm), pl)
    end
end

#for i=2:28
    #run(`mkdir linPlots/tract$i`)
#end



function fitter_l(X, age, w, D, lam1, lam2)
    (n, p) = size(X)
    b = Variable(1,p)
    beta = Variable(1,p)
    problem = minimize(quad_form(ones(n,1) * b + age * beta - X, diagm(vec(w))) + lam1*norm(D*b',1) + lam2*norm((beta),1))
    # Solve the problem by calling solve!
    out = solve!(problem, SCSSolver(max_iters = 100000, normalize = 0))
    #out = solve!(problem, ECOSSolver(maxit = 1000))
    return {"b"=>b.value, "beta"=>beta.value}
end
