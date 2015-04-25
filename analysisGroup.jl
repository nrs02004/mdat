using Gadfly
using Color
using Cairo
#using DataFrames
using Convex
using SCS
using ECOS
include("setup.jl")

tract_data = readdlm("All_Tracts_FA.csv",',')
age_data = readdlm("Subject_Age.csv",',')


### sort the data by age
ord = sortperm(age_data[:,1])
age_data = age_data[ord,1]

for tract_num = 1:28

    ## Removes the first and final tract values for each person & removes any people with NaN values
    X = tract_data[ord,(2:99) + 100*(tract_num-1)]
    keep = find(hasNaN(X).!=1)
    X = X[keep,:]; age = age_data[keep,1];

    junk = combine(age, X); age = junk["age"]; w = junk["w"]; X = junk["X"]; ## Combines people with identical ages, and computes an “average tract” over each unique age

    (n, p) = size(X)
    lam1 = 0.3; lam2s = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] ## lam1 is regularization param for smoothing, lam2 is regularization param for smoothing fits together

    D_a1 = form_D(n); D_age = DxInv(age,1); D_a2 = form_D(n-1);
    D0 = D_a1; #D_a2 * D_age * D_a1
    D1 = form_D(p); D2 = form_D(p-1); D3 = form_D(p-2);
    D = D3 * D2 * D1;

    fits = [fitter_group(X, w, D, D0, lam1, lam2) for lam2 in lam2s]

    colors = colormap("Purples", int(floor(1.5*n)));

    ## Makes plots and saves to a file
    plots = [plot([layer(x=1:p,y=fit[:,j], Geom.line, Theme(default_color=colors[j+floor(0.5*n)])) for j in 1:n]...) for fit in fits];

    for i=1:length(plots)
        draw(PDF("plots/tract$(tract_num)/plot$(i).pdf", 12cm, 8cm), plots[i])
    end
end

#for i=1:28
    #    run(`mkdir plots/tract$i`)
#end
