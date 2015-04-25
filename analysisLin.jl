using Gadfly
using Color
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
    lam1 = 0.4; lam2s = 0:10:80   ## lam1 is regularization param for smoothing, lam2 is regularization param for coefficient of age

    ## The following is used in fitting the smooth average profile ###
    D1 = form_D(p); D2 = form_D(p-1); D3 = form_D(p-2);
    D = D3 * D2 * D1;

    ## This fits our smooth profile + linear trend for penalties on age coefficient
    fits = [fitter_linear(X, age, w, D, lam1, lam2) for lam2 in lam2s]

    println("$tract_num")

    colors = colormap("Purples", int(floor(1.5*n)));

    ## Makes plots and saves to a file
    for i=1:length(fits)
        foo = ones(n,1) * fits[i]["b"] + age * fits[i]["beta"]
        pl = plot([layer(x=1:p,y=foo[j,:], Geom.line, Theme(default_color=colors[j+floor(0.5*n)])) for j in 1:n]...)
        draw(PDF("linPlots/tract$(tract_num)/plot$(i).pdf", 12cm, 8cm), pl)
    end
end

#for i=2:28
    #run(`mkdir linPlots/tract$i`)
#end
