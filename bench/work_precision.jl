# Work-precision data and solution traces for the figures, written to docs/data.
#
#     julia --project=. bench/work_precision.jl
#
# For each method and tolerance: correct digits at the end point against the
# Test Set reference (-log10 of the largest relative error), right-hand side
# evaluations, LU factorisations, steps and run time (best of three).

using StiffSolvers, Printf, LinearAlgebra

const OUT = joinpath(@__DIR__, "..", "docs", "data")
mkpath(OUT)

function digits(y, ref)
    -log10(maximum(abs.(y .- ref) ./ abs.(ref)))
end

function measure(io, name, prob, ref, method, tols, atolfac)
    for tol in tols
        sol = solve(prob, method; rtol=tol, atol=tol * atolfac, save_every=false, maxsteps=10^8)
        secs = minimum(@elapsed(solve(prob, method; rtol=tol, atol=tol * atolfac, save_every=false, maxsteps=10^8)) for _ in 1:3)
        s = sol.stats
        @printf(io, "%s,%s,%.0e,%.3f,%d,%d,%d,%d,%.6f\n", name, typeof(method), tol, digits(sol.u[end], ref),
                s.nf, s.nlu, s.naccept, s.nreject, secs)
        @printf("%-6s %-13s rtol %.0e: %5.2f digits, %9d f-evals, %8d steps, %8.4f s\n", name, typeof(method), tol,
                digits(sol.u[end], ref), s.nf, s.naccept, secs)
    end
end

open(joinpath(OUT, "work_precision.csv"), "w") do io
    println(io, "problem,method,rtol,digits,nf,nlu,steps,rejected,seconds")
    measure(io, "rober", robertson(), ROBERTSON_REF, RadauIIA5(), 10.0 .^ (-3:-1:-11), 1e-6)
    measure(io, "rober", robertson(), ROBERTSON_REF, Rosenbrock23(), 10.0 .^ (-2:-1:-8), 1e-6)
    measure(io, "vdpol", vanderpol(), VANDERPOL_REF, RadauIIA5(), 10.0 .^ (-3:-1:-11), 1.0)
    measure(io, "vdpol", vanderpol(), VANDERPOL_REF, Rosenbrock23(), 10.0 .^ (-2:-1:-8), 1.0)
    measure(io, "vdpol", vanderpol(), VANDERPOL_REF, DOPRI5(), 10.0 .^ (-3:-1:-7), 1.0)
end

# Traces: Robertson over 16 decades of time, Van der Pol with step sizes.
sol = solve(robertson(), RadauIIA5(); rtol=1e-8, atol=1e-14)
open(joinpath(OUT, "rober_trace.csv"), "w") do io
    println(io, "t,y1,y2,y3")
    for (t, y) in zip(sol.t, sol.u)
        @printf(io, "%.6e,%.10e,%.10e,%.10e\n", t, y[1], y[2], y[3])
    end
end
open(joinpath(OUT, "vdpol_steps.csv"), "w") do io
    println(io, "method,t,y1,h")
    for m in (RadauIIA5(), Rosenbrock23(), DOPRI5())
        s = solve(vanderpol(), m; rtol=1e-6, atol=1e-6, maxsteps=10^8)
        every = max(1, length(s.t) ÷ 2000)  # thin DOPRI5's million steps for plotting
        for k in 2:every:length(s.t)
            @printf(io, "%s,%.8e,%.8e,%.6e\n", typeof(m), s.t[k], s.u[k][1], s.t[k] - s.t[k-1])
        end
        @printf("%-13s Van der Pol at rtol 1e-6: %d steps\n", typeof(m), s.stats.naccept)
    end
end

# Orders of convergence with fixed steps on the rotation problem.
open(joinpath(OUT, "orders.csv"), "w") do io
    println(io, "method,h,error")
    for m in (DOPRI5(), Rosenbrock23(), RadauIIA5(), BackwardEuler()), k in 2:8
        h = 2.0^-k
        s = solve(rotation(), m; dt=h)
        @printf(io, "%s,%.8e,%.6e\n", typeof(m), h, hypot(s.u[end][1] - cos(1.0), s.u[end][2] - sin(1.0)))
    end
end

# Stability regions: |R(z)| on a grid of the complex plane (8 MB, not committed).

open(joinpath(OUT, "stability.csv"), "w") do io
    println(io, "method,re,im,absR")
    for m in (DOPRI5(), Rosenbrock23(), RadauIIA5(), BackwardEuler())
        for re in range(-12, 12; length=241), im in range(-12, 12; length=241)
            # A pole of R (z = 1 for backward Euler) makes the step's matrix singular.
            r = try
                abs(stability_function(m, complex(re, im)))
            catch e
                e isa LinearAlgebra.SingularException ? Inf : rethrow()
            end
            @printf(io, "%s,%.2f,%.2f,%.6e\n", typeof(m), re, im, r)
        end
    end
end
