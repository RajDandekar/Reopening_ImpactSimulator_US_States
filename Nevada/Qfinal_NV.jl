#Stage 1: 30,000 iterations would work.
#Stage 2: 30,000 iterations would work.
using MAT
using Plots
using Measures
using Flux
using DifferentialEquations
using DiffEqFlux
using LaTeXStrings
using StatsPlots
using Random
using JLD

######################## STAGE 1###############
vars = matread("/Users/Emma/Documents/School/2019-2020/Covid Urop/csse_covid_19_data/csse_covid_19_daily_reports/Rise_NV_Track.mat")

Random.seed!(60)

Infected = vars["NV_Infected_All"]
Recovered = vars["NV_Recovered_All"]
Dead= vars["NV_Dead_All"]
Time = vars["NV_Time"]

Infected = Infected - Recovered - Dead


ann = Chain(Dense(3,10,leakyrelu),  Dense(10,1))
p1,re = Flux.destructure(ann)
p2 = Float64[0.15, 0.013, 0.01]
p3 = [p1; p2]
ps = Flux.params(p3)

function QSIR(du, u, p, t)
    β = abs(p[52])
    γ = abs(p[53])
    δ = abs(p[54])
    #γ = abs(γ_parameter)
    #δ = abs(δ_parameter)
    un = [u[1]; u[2]; u[3]]
    NN1 = abs(re(p[1:51])(un)[1])
    du[1]=  - β*u[1]*(u[2])/u0[1]
    du[2] = β*u[1]*(u[2])/u0[1] - γ*u[2] - NN1*u[2]/u0[1]
    du[3] = γ*u[2] + δ*u[4]
    du[4] =  NN1*u[2]/u0[1] - δ*u[4]
end


u0 = Float64[3080000.0, 526, 10, 10]
tspan = (0, 110.0)
datasize = 110;


prob = ODEProblem(QSIR, u0, tspan, p3)
t = range(tspan[1],tspan[2],length=datasize)

sol = Array(concrete_solve(prob, Tsit5(),u0, p3, saveat=t))


function predict_adjoint() # Our 1-layer neural network
  Array(concrete_solve(prob,Tsit5(),u0,p3,saveat=t))
end


function loss_adjoint()
 prediction = predict_adjoint()
 loss = sum(abs2, log.(abs.(Infected[1:end])) .- log.(abs.(prediction[2, :] .+ prediction[4, :] ))) + (sum(abs2, log.(abs.(Recovered[1:end] + Dead[1:end]) ) .- log.(abs.(prediction[3, :] ))))
 #loss = sum(abs2, log.(abs.(Infected[1:end])) .- log.(abs.(prediction[2, :] .+ prediction[4,:])))
end

Loss = []
P1 = []
P2 = []
P3  =[]
anim = Animation()
datan = Iterators.repeated((), 120000)
opt = ADAM(0.01)
cb = function()
  display(loss_adjoint())
  global Loss = append!(Loss, loss_adjoint())
  global P1 = append!(P1, p3[52])
  global P2 = append!(P2, p3[53])
  global P3 = append!(P3, p3)
end


cb()


Flux.train!(loss_adjoint, ps, datan, opt, cb = cb)

L = findmin(Loss)
idx = L[2]
idx1 = (idx-1)*54 +1
idx2 = idx*54
p3n = P3[idx1: idx2]

prediction = Array(concrete_solve(prob,Tsit5(),u0,p3n,saveat=t))

S_NN_all_loss = prediction[1, :]
I_NN_all_loss = prediction[2, :]
R_NN_all_loss = prediction[3, :]
T_NN_all_loss = prediction[4, :]

 Q_parameter = zeros(Float64, length(S_NN_all_loss), 1)

for i = 1:length(S_NN_all_loss)
  Q_parameter[i] = abs(re(p3n[1:51])([S_NN_all_loss[i],I_NN_all_loss[i], R_NN_all_loss[i]])[1])
end

using JLD

save("QFinal_QuarHeatmap_NV_QSIR_Deadn.jld",  "β_parameter", p3n[52],"γ_parameter", p3n[53],"δ_parameter", p3n[54], "S_NN_all_loss", S_NN_all_loss, "I_NN_all_loss", I_NN_all_loss, "R_NN_all_loss", R_NN_all_loss,"t", t, "Parameters", p3,"Parameters_copy", p3n, "Loss", Loss)

##############STAGE 2#########################

vars = matread("/Users/Emma/Documents/School/2019-2020/Covid Urop/csse_covid_19_data/csse_covid_19_daily_reports/Rise_NV_Track.mat")

Random.seed!(50)

Infected = vars["NV_Infected_All"]
Recovered = vars["NV_Recovered_All"]
Dead= vars["NV_Dead_All"]
Time = vars["NV_Time"]

Infected = Infected - Recovered - Dead


ann = Chain(Dense(3,10,leakyrelu),  Dense(10,1))
p1,re = Flux.destructure(ann)
p2 = Float64[0.15, 0.013, 0.01]
p3 = [p1; p2]
ps = Flux.params(p3)

D = load("QFinal_QuarHeatmap_NV_QSIR_Deadn.jld")
γ_parameter = D["γ_parameter"]
δ_parameter = D["δ_parameter"]


function QSIR(du, u, p, t)
    β = abs(p[52])
    #γ = abs(p[53])
    #δ = abs(p[54])
    γ = abs(γ_parameter)
    δ = abs(δ_parameter)
    un = [u[1]; u[2]; u[3]]
    NN1 = abs(re(p[1:51])(un)[1])
    du[1]=  - β*u[1]*(u[2])/u0[1]
    du[2] = β*u[1]*(u[2])/u0[1] - γ*u[2] - NN1*u[2]/u0[1]
    du[3] = γ*u[2] + δ*u[4]
    du[4] =  NN1*u[2]/u0[1] - δ*u[4]
end


u0 = Float64[3080000.0, 526, 10, 10]
tspan = (0, 110.0)
datasize = 110;


prob = ODEProblem(QSIR, u0, tspan, p3)
t = range(tspan[1],tspan[2],length=datasize)

sol = Array(concrete_solve(prob, Tsit5(),u0, p3, saveat=t))


function predict_adjoint() # Our 1-layer neural network
  Array(concrete_solve(prob,Tsit5(),u0,p3,saveat=t))
end


function loss_adjoint()
 prediction = predict_adjoint()
 #loss = sum(abs2, log.(abs.(Infected[1:end])) .- log.(abs.(prediction[2, :] .+ prediction[4, :] ))) + (sum(abs2, log.(abs.(Recovered[1:end] + Dead[1:end]) ) .- log.(abs.(prediction[3, :] ))))
 loss = sum(abs2, log.(abs.(Infected[1:end])) .- log.(abs.(prediction[2, :] .+ prediction[4,:])))
end

Loss = []
P1 = []
P2 = []
P3  =[]
anim = Animation()
datan = Iterators.repeated((), 60000)
opt = ADAM(0.01)
cb = function()
  display(loss_adjoint())
  global Loss = append!(Loss, loss_adjoint())
  global P1 = append!(P1, p3[52])
  global P2 = append!(P2, p3[53])
  global P3 = append!(P3, p3)
end


cb()


Flux.train!(loss_adjoint, ps, datan, opt, cb = cb)

#gif(anim,"Quar_Dead_NV.gif", fps=15)
L = findmin(Loss)
idx = L[2]
idx1 = (idx-1)*54 +1
idx2 = idx*54
p3n = P3[idx1: idx2]

prediction = Array(concrete_solve(prob,Tsit5(),u0,p3n,saveat=t))

S_NN_all_loss = prediction[1, :]
I_NN_all_loss = prediction[2, :]
R_NN_all_loss = prediction[3, :]
T_NN_all_loss = prediction[4, :]

 Q_parameter = zeros(Float64, length(S_NN_all_loss), 1)

for i = 1:length(S_NN_all_loss)
  Q_parameter[i] = abs(re(p3n[1:51])([S_NN_all_loss[i],I_NN_all_loss[i], R_NN_all_loss[i]])[1])
end

#Infected and recovered count
using Plots

bar(Infected',alpha=0.5,label="Data: Infected",color="red")
plot!(t, prediction[2, :] .+ prediction[4,:], xaxis = "Days post 500 infected", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 4, ylims = (0, 20000), foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
bar!(Recovered' + Dead',alpha=0.5,xrotation=60,label="Data: Recovered + Dead", color="blue")
plot!(t, prediction[3, :], ylims = (-0.03*maximum(Recovered + Dead),1.5*maximum(Infected)), right_margin = 5mm, xaxis = "Days post 500 infected", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman", legendfontsize = 1))

savefig("QFinal_Quar_Stage2_NV_1d.pdf")



#Reproduction number
pyplot()
Reff = abs(p3n[52]) ./ (abs(γ_parameter) .+ abs(δ_parameter) .+ Q_parameter/u0[1])
Transition = findall(x -> x <1, Reff)[1]
plot()
plot([0, Transition[1]], [80, 80],fill=(0,:lightpink), markeralpha=0, label = "")
plot!([Transition[1], datasize], [80, 80],fill=(0,:aliceblue), markeralpha=0, label = "")
scatter!(t[5:end], Reff[5:end], xlims = (0, datasize), ylims = (0.5, 5), xlabel = "Days post 500 infected", label = "Effective reproduction number", legend = :topright, color = :black, framestyle = :box, grid =:off, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, left_margin = 5mm, bottom_margin= 5mm)
f(x) = 1
plot!(f, xlims = (0, datasize), color = :blue, linewidth = 3, label = L"R_{t} = 1")

Plots.savefig("QFinal_Quar_Stage2_NV_2d.pdf")

#Quarantine strength
pyplot()
plot([0, Transition[1]], [80, 80],fill=(0,:lightpink), markeralpha=0, label = "")
plot!([Transition[1], datasize], [80, 80],fill=(0,:aliceblue), markeralpha=0, label = "")
scatter!(t,Q_parameter/u0[1], xlims = (0, datasize), ylims = (0,1), xlabel = "Days post 500 infected", ylabel = "Q(t)", label = "Quarantine strength",color = :black, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

Plots.savefig("QFinal_Quar_Stage2_NV_3d.pdf")


Q_NV = ((Q_parameter[30] - Q_parameter[1]) /u0[1])

save("QFinal_QuarHeatmap_NV_QSIR_Deadn_Stage2.jld", "Q_NV", Q_NV, "prediction", prediction, "Q_parameter", Q_parameter, "Reff", Reff, "Transition", Transition, "β_parameter", p3n[52],"γ_parameter", γ_parameter,"δ_parameter", δ_parameter, "S_NN_all_loss", S_NN_all_loss, "I_NN_all_loss", I_NN_all_loss, "R_NN_all_loss", R_NN_all_loss,"t", t, "Parameters", p3,"Parameters_copy", p3n, "Loss", Loss)


########CLEAN PLOTS###########
D = load("QFinal_QuarHeatmap_NV_QSIR_Deadn_Stage2.jld")
prediction = D["prediction"]
Reff = D["Reff"]
Transition = D["Transition"]
Q_parameter = D["Q_parameter"]

vars = matread("/Users/Emma/Documents/School/2019-2020/Covid Urop/csse_covid_19_data/csse_covid_19_daily_reports/Rise_NV_Track.mat")
Infected = vars["NV_Infected_All"]
Recovered = vars["NV_Recovered_All"]
Dead= vars["NV_Dead_All"]
Time = vars["NV_Time"]

Infected = Infected - Recovered - Dead

u0 = Float64[3080000.0, 526, 10, 10]
tspan = (0, 110.0)
datasize = 110;

t = range(tspan[1],tspan[2],length=datasize)

bar(Infected',alpha=0.5,label="Data: Infected",color="red")
plot!(t, prediction[2, :] .+ prediction[4, :] , xaxis = "Days post 500 infected", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
bar!(Recovered' + Dead',alpha=0.5,xrotation=60,label="Data: Recovered", color="blue")
plot!(t, prediction[3, :], ylims = (-0.05*maximum(Recovered + Dead),1.5*maximum(Infected)), right_margin = 5mm, xaxis = "Days post 500 infected", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman", legendfontsize = 1))

savefig("Clean_QFinal_Quar_Stage2_NV_1dn.pdf")

using LaTeXStrings

#plot([0, Transition[1]], [80, 80],fill=(0,:lightpink), markeralpha=0, label = "")
#plot!([Transition[1], datasize], [80, 80],fill=(0,:aliceblue), markeralpha=0, label = "")
scatter(t[5:end], Reff[5:end], xlims = (0, datasize), ylims = (0.5, 5), xlabel = "Days post 500 infected", label = string("Covid spread parameter ", L"C_{p}"), legend = :topright, color = :black, framestyle = :box, grid =:off, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, left_margin = 5mm, bottom_margin= 5mm)
f(x) = 1
plot!(f, xlims = (0,datasize), color = :blue, linewidth = 3, label = L"C_{p} = 1")

Plots.savefig("Clean_QFinal_Quar_Stage2_NV_2d.pdf")

#plot([0, Transition[1]], [80, 80],fill=(0,:lightpink), markeralpha=0, label = "")
#plot!([Transition[1], datasize], [80, 80],fill=(0,:aliceblue), markeralpha=0, label = "")
scatter(t,Q_parameter/u0[1], xlims = (0, datasize), ylims = (0,1), xlabel = "Days post 500 infected", ylabel = "Q(t)", label = "Quarantine strength",color = :black, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

D1 = diff(Q_parameter[1:end], dims = 1)
D2 = diff(D1, dims = 1)
#Transitionn = findall(x -> x <0, D2)[2]

plot!([43-0.01,43+0.01],[0.0, 0.6],lw=3,color=:green,label="Stay-at-home order expires",linestyle = :dash)

#plot!([15-0.01,15+0.01],[0.0, 0.6],lw=3,color=:red,label="Inflection point in learnt Q(t)",linestyle = :dash)

Plots.savefig("Clean_QFinal_Quar_Stage2_NV_3dn.pdf")

#######WHAT IF NO REOPENING##############
D = load("QFinal_QuarHeatmap_NV_QSIR_Deadn_Stage2.jld")
prediction = D["prediction"]
Reff = D["Reff"]
Transition = D["Transition"]
Q_parameter = D["Q_parameter"]
p3n = D["Parameters_copy"]
β_parameter = D["β_parameter"]
γ_parameter = D["γ_parameter"]
δ_parameter = D["δ_parameter"]

vars = matread("/Users/Emma/Documents/School/2019-2020/Covid Urop/csse_covid_19_data/csse_covid_19_daily_reports/Rise_NV_Track.mat")
Infected = vars["NV_Infected_All"]
Recovered = vars["NV_Recovered_All"]
Dead= vars["NV_Dead_All"]
Time = vars["NV_Time"]

Infected = Infected - Recovered - Dead

u0 = Float64[3080000.0, 526, 10, 10]
tspan = (0, 110.0)
datasize = 110;

t = range(tspan[1],tspan[2],length=datasize)

pyplot()
bar(Infected',alpha=0.6,label="Data: Infected",color="red")
plot!(t, prediction[2, :] .+ prediction[4, :] , xaxis = "Days post 500 infected", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
bar!(Recovered' + Dead',alpha=0.5,xrotation=60,label="Data: Recovered", color="green")
plot!(t, prediction[3, :], ylims = (-1.25*maximum(Recovered + Dead),1.5*maximum(Infected)), right_margin = 5mm, xaxis = "Days post 500 infected", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :darkgreen, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman", legendfontsize = 1))

savefig("NV_reopen_1.pdf")

reopening_day = 43
Q_parameter_noreopen = vcat(Q_parameter[1:reopening_day], Q_parameter[reopening_day]*ones(length(Q_parameter)-reopening_day, 1))

scatter(t,Q_parameter/u0[1], xlims = (0, datasize), ylims = (0,1), xlabel = "Days post 500 infected", ylabel = "Q(t)", label = "Actual Quarantine Strength: Reopening",color = :red, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
scatter!(t,Q_parameter_noreopen/u0[1], xlims = (0, datasize), ylims = (0,1), xlabel = "Days post 500 infected", ylabel = "Q(t)", label = "Simulated Quarantine Strength: No reopening",color = :blue, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
plot!([reopening_day-0.01,reopening_day+0.01],[0.0, 0.6],lw=3,color=:green,label="Stay-at-home order expires",linestyle = :dash)

savefig("NV_reopen_2.pdf")

function QSIR_NoReopening(du, u, p, t)
    index = Int(floor(t))
    β = abs(β_parameter)
    γ = abs(γ_parameter)
    δ = abs(δ_parameter)
    un = [u[1]; u[2]; u[3]]
    NN1 = Q_parameter_noreopen[index]
    du[1]=  - β*u[1]*(u[2])/u0[1]
    du[2] = β*u[1]*(u[2])/u0[1] - γ*u[2] - NN1*u[2]/u0[1]
    du[3] = γ*u[2] + δ*u[4]
    du[4] =  NN1*u[2]/u0[1] - δ*u[4]
end

tspan = (1, 110.0)
prob = ODEProblem(QSIR_NoReopening, u0, tspan, p3n)
t = range(tspan[1],tspan[2],length=datasize)

sol = Array(concrete_solve(prob, Tsit5(),u0, p3n, saveat=t))

prediction_noreopen= Array(concrete_solve(prob,Tsit5(),u0,p3n,saveat=t))

# bar(Infected',alpha=0.5,label="Data: Infected",color="red")
# plot!(t, prediction_noreopen[2, :] .+ prediction_noreopen[4, :] , xaxis = "Days post 500 infected", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
# bar!(Recovered' + Dead',alpha=0.5,xrotation=60,label="Data: Recovered", color="blue")
# plot!(t, prediction_noreopen[3, :], ylims = (-0.05*maximum(Recovered + Dead),1.75*maximum(Infected)), right_margin = 5mm, xaxis = "Days post 500 infected", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman", legendfontsize = 1))
# savefig("NV_noreopen_count.pdf")

bar(Infected',alpha=0.6,label="Infected Data: Reopening",color="red")
bar!(prediction_noreopen[2, :] .+ prediction_noreopen[4, :],alpha=0.5,xrotation=60,label="Infected Prediction: No reopening", color="blue",ylims = (-1.25*maximum(Recovered + Dead),1.5*maximum(Infected)), right_margin = 5mm, xaxis = "Days post 500 infected",legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off,foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman", legendfontsize = 12),legendfontsize = 12)

savefig("NV_reopen_3.pdf")

Infected_cases_reduced = ((prediction[2, :] .+ prediction[4, :] .+ prediction[3, :]) .- (prediction_noreopen[2,:] .+ prediction_noreopen[4,:] .+ prediction_noreopen[3,:]))
Infected_count_saved = Infected_cases_reduced[end]
println("Cumulative case reduction: ", Infected_count_saved)
Total_infected_cases = prediction[2, end] + prediction[3, end] + prediction[4, end]
println("Percent decrease: ", Infected_count_saved/Total_infected_cases)

min_Q = minimum(Q_parameter[reopening_day:datasize])/u0[1]
reopening_Q = Q_parameter[reopening_day]/u0[1]
println("Q(t) at reopening: ", reopening_Q)
println("Minimum Q(t) after reopening: ", min_Q)
println("Percent decrease of Q(t): ", (reopening_Q-min_Q)/reopening_Q)
