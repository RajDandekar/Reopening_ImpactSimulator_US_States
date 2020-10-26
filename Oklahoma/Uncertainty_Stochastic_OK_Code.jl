using DataInterpolations, Plots
using DiffEqBiological
using DiffEqBase, OrdinaryDiffEq
using DifferentialEquations
using Plots, Measures, StatsPlots
using Flux, DiffEqFlux
using JLD
using DifferentialEquations.EnsembleAnalysis
using Statistics
using MAT
using LsqFit

D = load("/Users/Emma/QFinal_QuarHeatmap_OK_QSIR_Deadn_Stage2.jld")
prediction = D["prediction"]
Reff = D["Reff"]
Transition = D["Transition"]
p3n = D["Parameters_copy"]
Q_parameter = D["Q_parameter"]
β_parameter = D["β_parameter"]
γ_parameter = D["γ_parameter"]
δ_parameter = D["δ_parameter"]


vars = matread("/Users/Emma/Documents/School/2019-2020/Covid Urop/csse_covid_19_data/csse_covid_19_daily_reports/Rise_OK_Track.mat")
Infected = vars["OK_Infected_All"]
Recovered = vars["OK_Recovered_All"]
Dead= vars["OK_Dead_All"]
Time = vars["OK_Time"]

Infected = Infected - Recovered - Dead

u0 = Float64[3960000.0, 545, 23, 10]
tspan = (0, 106.0)
datasize = 106;

t = range(tspan[1],tspan[2],length=datasize)

func = QuadraticInterpolation(Q_parameter[1:end]/u0[1],Array(t))


##Check the fit. Here, I am converting the array Q_parameter to a function since we need that when we define the reaction network
scatter(Q_parameter/u0[1])

scatter!(func)

##Defining the reaction network
rn = @reaction_network begin
    β, S + I --> 2I
    γ, I --> R
    func(t), I --> Q
    δ, Q --> R
end β γ δ

p   = [β_parameter/u0[1], abs(γ_parameter), abs(δ_parameter)]

#######ODE PROBLEM: CHECK IF THIS MATCHES WHAT YOU GOT WHILE RUNNING EARLIER MODEL. IT SHOULD MATCH IN THEORY########

op  = ODEProblem(rn, u0, tspan, p)

prediction_Catalystn = Array(concrete_solve(op, Tsit5(),u0,p,saveat=t))

bar(Infected',alpha=0.5,label="Data: Infected",color="red")
plot!(t, prediction_Catalystn[2, :] .+ prediction_Catalystn[4, :] , xaxis = "Days post 500 infected", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
bar!(Recovered' + Dead',alpha=0.5,xrotation=60,label="Data: Recovered", color="blue")
plot!(t, prediction_Catalystn[3, :], ylims = (-0.05*maximum(Recovered + Dead),1.5*maximum(Infected)), right_margin = 5mm, xaxis = "Days post 500 infected", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman", legendfontsize = 1))

###THIS IS THE ADVANTAGE OF USING A REACTION NETWORK. WE CAN CONVERT THE PROBLEM TO A STOCHASTIC ONE.
### SDE REFERS TO STOCHASTIC DIFFERENTIAL EQUATION
####SDE PROBLEM WITH GAUSSIAN WHITE NOISE####
SDE = SDEProblem(rn, u0, tspan, p)

###SDE ENSEMBLE PROBLEM: RUN FOR 1000 TRAJETORIES TO GET UNCERTAINTY######
ensembleprob = EnsembleProblem(SDE)

sol = solve(ensembleprob,LambaEM(),saveat=t,trajectories=1000)

summ = EnsembleSummary(sol)
pyplot() # Note that plotly does not support ribbon plots


mean_value  = timeseries_steps_mean(sol)
low_quantile = timeseries_steps_quantile(sol, 0.05)
high_quantile = timeseries_steps_quantile(sol, 0.95)

Mean_Infected = mean_value[2,:] .+ mean_value[4,:]
Low_quantile_Infected = low_quantile[2,:] .+ low_quantile[4,:]
High_quantile_Infected = high_quantile[2,:] .+ high_quantile[4,:]

#CHECK SDE PLOT###
bar(Infected',alpha=0.2,label="Data: Infected",color="red")
plot!(t, Mean_Infected, ribbon=(Mean_Infected .- Low_quantile_Infected,High_quantile_Infected .- Mean_Infected), fillalpha=0.6, c=:red, lab="Chemical Langevin SDE", linewidth = 3)
plot!(t, prediction[2, :] .+ prediction[4, :] , xaxis = "Days post 500 infected", label = "ODE model", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :black, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

bar!(Recovered' + Dead',alpha=0.2,xrotation=60,label="Data: Recovered", color="blue")
plot!(summ,fillalpha=0.5, color = :blue, idxs=3, legend = :topleft, lab="Chemical Langevin SDE", framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
plot!(t, prediction[3, :], ylims = (-0.05*maximum(Recovered + Dead),1.5*maximum(Infected)), right_margin = 5mm, xaxis = "Days post 500 infected", label = "ODE model ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :black, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman", legendfontsize = 1))


#### WHAT IF NO REOPENING: SAME PROCESS FOR NO REOPENING##########
reopening_day = 24
Q_parameter_noreopen = vcat(Q_parameter[1:reopening_day], Q_parameter[reopening_day]*ones(length(Q_parameter)-reopening_day, 1))

func2 = QuadraticInterpolation(Q_parameter_noreopen[1:end]/u0[1],Array(t))


##Check the fit
scatter(Q_parameter_noreopen/u0[1])

scatter!(func2)

##Defining the reaction network
rn2 = @reaction_network begin
    β, S + I --> 2I
    γ, I --> R
    func2(t), I --> Q
    δ, Q --> R
end β γ δ

p   = [β_parameter/u0[1], abs(γ_parameter), abs(δ_parameter)]           # [α,β]
#######ODE PROBLEM########

op  = ODEProblem(rn2, u0, tspan, p)

prediction_Catalystn2 = Array(concrete_solve(op, Tsit5(),u0,p,saveat=t))

bar(Infected',alpha=0.5,label="Data: Infected",color="red")
plot!(t, prediction_Catalystn2[2, :] .+ prediction_Catalystn2[4, :] , xaxis = "Days post 500 infected", label = "No reopening: Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
bar!(Recovered' + Dead',alpha=0.5,xrotation=60,label="Data: Recovered", color="blue")
plot!(t, prediction_Catalystn2[3, :], ylims = (-0.05*maximum(Recovered + Dead),1.5*maximum(Infected)), right_margin = 5mm, xaxis = "Days post 500 infected", label = "No reopening: Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman", legendfontsize = 1))


####SDE PROBLEM WITH GAUSSIAN WHITE NOISE####
SDE = SDEProblem(rn2, u0, tspan, p)
#prediction_Catalyst_SDE = Array(concrete_solve(SDE, LambaEM(),u0, p, saveat=t))

###SDE ENSEMBLE PROBLEM######
ensembleprob = EnsembleProblem(SDE)

sol = solve(ensembleprob,LambaEM(),saveat=t,trajectories=1000)

summ2 = EnsembleSummary(sol)
pyplot() # Note that plotly does not support ribbon plots


mean_value2  = timeseries_steps_mean(sol)
low_quantile2 = timeseries_steps_quantile(sol, 0.05)
high_quantile2 = timeseries_steps_quantile(sol, 0.95)

Mean_Infected2 = mean_value2[2,:] .+ mean_value2[4,:]
Low_quantile_Infected2 = low_quantile2[2,:] .+ low_quantile2[4,:]
High_quantile_Infected2 = high_quantile2[2,:] .+ high_quantile2[4,:]

###CHECK SDE PLOTS########
bar(Infected',alpha=0.2,label="Data: Infected",color="red")
plot!(t, Mean_Infected2, ribbon=(Mean_Infected2 .- Low_quantile_Infected2,High_quantile_Infected2 .- Mean_Infected2), fillalpha=0.6, c=:red, lab="No reopening: Chemical Langevin SDE", linewidth = 3)
plot!(t, prediction[2, :] .+ prediction[4, :] , xaxis = "Days post 500 infected", label = "No reopening: ODE model", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :black, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

bar!(Recovered' + Dead',alpha=0.2,xrotation=60,label="Data: Recovered", color="blue")
plot!(summ2,fillalpha=0.5, color = :blue, idxs=3, legend = :topleft, lab="No reopening:  Chemical Langevin SDE", framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
plot!(t, prediction[3, :], ylims = (-0.05*maximum(Recovered + Dead),1.5*maximum(Infected)), right_margin = 5mm, xaxis = "Days post 500 infected", label = "No reopening: ODE model ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :black, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman", legendfontsize = 1))

########CLEAN PLOTS: PLOTS THAT WILL GO IN THE PAPER FOR NOW#######
bar(Infected', xaxis = "Days post 500 infected",alpha=0.4,label="Data: Infected",color="red")
plot!(t, Mean_Infected, ribbon=(Mean_Infected .- Low_quantile_Infected,High_quantile_Infected .- Mean_Infected), fillalpha=0.6, c=:red, lab="Prediction", linewidth = 3, ylims = (-0.04*maximum(Recovered + Dead),3.75*maximum(Infected)))

bar!(Recovered' + Dead',alpha=0.4,xrotation=60,label="Data: Recovered", color="green")
plot!(summ,fillalpha=0.6, color = :green, idxs=3, legend = :topleft, lab="Prediction", framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm, right_margin = 5mm, grid = :off, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)


savefig("OK_CLE_1.pdf")

bar(Infected', xaxis = "Days post 500 infected",alpha=0.4,label="Infected Data: Reopening",color="red", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm, right_margin = 5mm, grid = :off, xrotation = 60, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
bar!(Mean_Infected2, alpha=0.4, c=:blue, label="")
plot!(t, Mean_Infected2, ribbon=(Mean_Infected2 .- Low_quantile_Infected2,High_quantile_Infected2 .- Mean_Infected2), fillalpha=0.6, c=:blue, lab="Infected Prediction: No reopening", linewidth = 3, ylims = (-0.02*maximum(Recovered + Dead),1.5*maximum(Infected)))

savefig("OK_CLE_2.pdf")

Actual_infected_cases = Infected[end]+Recovered[end]+Dead[end]
Low_infected_cases_reduced = Actual_infected_cases - (high_quantile2[2, end] + high_quantile2[4, end] + high_quantile2[3, end])
Mean_infected_cases_reduced = Actual_infected_cases - (mean_value2[2, end] + mean_value2[4, end] + mean_value2[3, end])
High_infected_cases_reduced = Actual_infected_cases - (low_quantile2[2, end] + low_quantile2[4, end] + low_quantile2[3, end])
println("Low case reduction: ", Low_infected_cases_reduced)
println("Mean case reduction: ", Mean_infected_cases_reduced)
println("High case reduction: ", High_infected_cases_reduced)
println("Low percent decrease: ", Low_infected_cases_reduced/Actual_infected_cases)
println("Mean percent decrease: ", Mean_infected_cases_reduced/Actual_infected_cases)
println("High percent decrease: ", High_infected_cases_reduced/Actual_infected_cases)
