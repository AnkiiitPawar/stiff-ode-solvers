"""
    StiffSolvers

Four one-step ODE solvers written from scratch with only the standard library:

- `DOPRI5`: explicit Dormand-Prince 5(4), the non-stiff reference;
- `Rosenbrock23`: the L-stable linearly implicit 2(3) pair of MATLAB's ode23s;
- `RadauIIA5`: the 3-stage Radau IIA collocation method of order 5, solved by
  simplified Newton iteration, with Hairer and Wanner's embedded error estimate;
- `BackwardEuler`: fixed steps, as the simplest L-stable method.

`solve(prob, method; rtol, atol)` adapts the step to the tolerances;
`solve(prob, method; dt)` takes fixed steps.
"""
module StiffSolvers

using LinearAlgebra

export ODEProblem, Solution, solve, DOPRI5, Rosenbrock23, RadauIIA5, BackwardEuler, stability_function
export robertson, ROBERTSON_REF, vanderpol, VANDERPOL_REF, prothero_robinson, rotation

"""
    ODEProblem(f, u0, tspan; jac=nothing)

y' = f(y, t) on `tspan` from `u0`. `jac(y, t)` returns the Jacobian matrix;
without it, the implicit methods use forward differences.
"""
struct ODEProblem{F,J,T}
    f::F
    u0::Vector{T}
    tspan::Tuple{Float64,Float64}
    jac::J
end
ODEProblem(f, u0, tspan; jac=nothing) = ODEProblem(f, collect(u0), (Float64(tspan[1]), Float64(tspan[2])), jac)

mutable struct Stats
    nf::Int       # right-hand side evaluations (finite-difference Jacobians included)
    njac::Int     # Jacobian evaluations
    nlu::Int      # LU factorisations
    naccept::Int  # accepted steps
    nreject::Int  # rejected steps
    nnewton::Int  # Newton iterations
end
Stats() = Stats(0, 0, 0, 0, 0, 0)

struct Solution{T}
    t::Vector{Float64}
    u::Vector{Vector{T}}
    stats::Stats
    retcode::Symbol  # :success, :maxsteps or :dtmin
end

abstract type Method end
struct DOPRI5 <: Method end
struct Rosenbrock23 <: Method end
struct RadauIIA5 <: Method end
struct BackwardEuler <: Method end

order(::DOPRI5) = 5
order(::Rosenbrock23) = 2
order(::RadauIIA5) = 5
order(::BackwardEuler) = 1
# Order of the local error estimate, which sets the step-size exponent.
estimate_order(::DOPRI5) = 5
estimate_order(::Rosenbrock23) = 3
estimate_order(::RadauIIA5) = 4

f!(prob, st, y, t) = (st.nf += 1; prob.f(y, t))

function jacobian(prob, st, y, t)
    st.njac += 1
    prob.jac !== nothing && return prob.jac(y, t)
    f0 = f!(prob, st, y, t)
    n = length(y)
    J = zeros(eltype(f0), n, n)
    for j in 1:n
        δ = sqrt(eps(Float64)) * max(abs(y[j]), 1e-5)
        yj = copy(y)
        yj[j] += δ
        J[:, j] = (f!(prob, st, yj, t) - f0) / δ
    end
    return J
end

# Scaled RMS norm of an error estimate (Hairer, Norsett and Wanner, II.4).
function errnorm(err, y0, y1, rtol, atol)
    s = 0.0
    for i in eachindex(err)
        sc = atol + rtol * max(abs(y0[i]), abs(y1[i]))
        s += abs2(err[i] / sc)
    end
    return sqrt(s / length(err))
end

include("dopri5.jl")
include("rosenbrock23.jl")
include("radau.jl")
include("backward_euler.jl")
include("problems.jl")

"""
    solve(prob, method; rtol=1e-6, atol=1e-6, dt=nothing, h0=nothing, maxsteps=10^7, save_every=true)

With `dt`, take fixed steps of that size. Otherwise choose the steps so the
local error estimate stays within `atol + rtol*|y|`.
"""
function solve(prob::ODEProblem, method::Method; rtol=1e-6, atol=1e-6, dt=nothing, h0=nothing,
               maxsteps=10^7, save_every=true)
    st = Stats()
    t0, tf = prob.tspan
    y = copy(prob.u0)
    ts, us = [t0], [copy(y)]
    cache = init_cache(method, prob, y)
    if dt !== nothing || method isa BackwardEuler
        h = dt === nothing ? error("BackwardEuler needs a fixed step dt") : Float64(dt)
        t = t0
        while t < tf - 1e-12 * abs(tf)
            hs = min(h, tf - t)
            y, _, ok = step(method, cache, prob, st, y, t, hs, Inf, Inf, true)
            ok || return Solution(ts, us, st, :newton)
            t += hs
            st.naccept += 1
            if save_every || t >= tf - 1e-12 * abs(tf)
                push!(ts, t)
                push!(us, copy(y))
            end
        end
        return Solution(ts, us, st, :success)
    end

    t = t0
    h = h0 === nothing ? initial_step(prob, st, y, t, rtol, atol, estimate_order(method)) : Float64(h0)
    q = estimate_order(method)
    fresh = true  # recompute the Jacobian at the start of the step
    for _ in 1:maxsteps
        h = min(h, tf - t)
        h < 1e-14 * max(abs(t), 1.0) && return Solution(ts, us, st, :dtmin)
        ynew, err, ok = step(method, cache, prob, st, y, t, h, rtol, atol, fresh)
        if !ok  # Newton failed to converge: shrink the step and retry
            st.nreject += 1
            h /= 2
            fresh = true
            continue
        end
        fac = 0.9 * min(1.0, newton_safety(method, cache))
        if err <= 1
            t += h
            y = ynew
            st.naccept += 1
            if save_every || t >= tf
                push!(ts, t)
                push!(us, copy(y))
            end
            t >= tf && return Solution(ts, us, st, :success)
            h *= clamp(fac * err^(-1 / q), 0.2, 5.0)
            fresh = true
        else
            st.nreject += 1
            h *= clamp(fac * err^(-1 / q), 0.2, 1.0)
            fresh = false  # same point: the Jacobian is still valid
        end
    end
    return Solution(ts, us, st, :maxsteps)
end

newton_safety(::Method, cache) = 1.0

# Initial step from Hairer, Norsett and Wanner (II.4): make an explicit Euler
# step's estimated error about the tolerance.
function initial_step(prob, st, y, t, rtol, atol, q)
    f0 = f!(prob, st, y, t)
    sc = atol .+ rtol .* abs.(y)
    d0, d1 = norm(y ./ sc) / sqrt(length(y)), norm(f0 ./ sc) / sqrt(length(y))
    h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1
    y1 = y + h0 * f0
    f1 = f!(prob, st, y1, t + h0)
    d2 = norm((f1 - f0) ./ sc) / sqrt(length(y)) / h0
    h1 = max(d1, d2) <= 1e-15 ? max(1e-6, h0 * 1e-3) : (0.01 / max(d1, d2))^(1 / (q + 1))
    return min(100 * h0, h1, prob.tspan[2] - t)
end

"""
    stability_function(method, z)

R(z) for the method: the factor one step of size 1 applies to the solution of
y' = z y.
"""
function stability_function(method::Method, z::Number)
    zc = ComplexF64(z)
    prob = ODEProblem((y, t) -> zc .* y, [one(zc)], (0.0, 1.0); jac=(y, t) -> fill(zc, 1, 1))
    cache = init_cache(method, prob, prob.u0)
    y, _, _ = step(method, cache, prob, Stats(), prob.u0, 0.0, 1.0, Inf, Inf, true)
    return y[1]
end

end
