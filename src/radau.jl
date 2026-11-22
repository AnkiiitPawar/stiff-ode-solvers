# Radau IIA with three stages (order 5, stage order 3, L-stable), after
# Hairer and Wanner, Solving Ordinary Differential Equations II, IV.8.
#
# The stage increments z_i = Y_i - y0 satisfy z = h (A ⊗ I) F(y0 + z), solved
# by simplified Newton iteration with the matrix I - h (A ⊗ J). That matrix is
# 3n x 3n and factorised directly; RADAU5 instead diagonalises A to work with
# one real and one complex n x n system, which only matters for large n.
#
# Error estimate: an embedded method of order 3 that adds an explicit term
# gamma0 h f(y0), where 1/gamma0 is the real eigenvalue of A^-1. Its
# difference from the Radau solution is filtered through (I - h gamma0 J)^-1,
# which keeps the estimate bounded for very stiff components.

const SQ6 = sqrt(6.0)
const RADAU_C = [(4 - SQ6) / 10, (4 + SQ6) / 10, 1.0]
const RADAU_A = [(88 - 7SQ6)/360      (296 - 169SQ6)/1800  (-2 + 3SQ6)/225
                 (296 + 169SQ6)/1800  (88 + 7SQ6)/360      (-2 - 3SQ6)/225
                 (16 - SQ6)/36        (16 + SQ6)/36        1/9]
const RADAU_B = RADAU_A[3, :]

# gamma0 = 1 / (real eigenvalue of A^-1); the embedded weights bhat satisfy the
# order-3 conditions gamma0 + sum(bhat) = 1, bhat.c = 1/2, bhat.c^2 = 1/3; and
# y1 - yhat1 = sum(e_i z_i) - h gamma0 f(y0) with e = A^-T (b - bhat).
const RADAU_GAMMA0 = 1 / real(only(filter(x -> abs(imag(x)) < 1e-12, eigvals(inv(RADAU_A)))))
const RADAU_BHAT = [1.0 1.0 1.0; RADAU_C'; (RADAU_C .^ 2)'] \ [1 - RADAU_GAMMA0, 1 / 2, 1 / 3]
const RADAU_E = inv(RADAU_A)' * (RADAU_B - RADAU_BHAT)

const NEWTON_MAX = 7

mutable struct RadauCache{M}
    J::M
    iters::Int
end

init_cache(::RadauIIA5, prob, y) = RadauCache(zeros(eltype(y), length(y), length(y)), 1)

# Fewer Newton iterations last step: closer to the full safety factor (RADAU5's rule).
newton_safety(::RadauIIA5, c::RadauCache) = (1 + 2NEWTON_MAX) / (c.iters + 2NEWTON_MAX)

function step(::RadauIIA5, c::RadauCache, prob, st, y, t, h, rtol, atol, fresh)
    n = length(y)
    T = eltype(y)
    fresh && (c.J = jacobian(prob, st, y, t))
    M = lu(I - h .* kron(RADAU_A, c.J))
    st.nlu += 1
    # Newton tolerance: a small fraction of the error tolerance, or tight
    # absolute and relative limits when the steps are fixed.
    fixed = isinf(rtol)
    sc = fixed ? fill(1e-15, n) .+ 1e-13 .* abs.(y) : atol .+ rtol .* abs.(y)
    κ = fixed ? 1.0 : 0.03
    z = zeros(T, 3n)
    nrm_old, η = Inf, 1.0
    converged = false
    for it in 1:(fixed ? 50 : NEWTON_MAX)
        st.nnewton += 1
        F = similar(z)
        for i in 1:3
            F[(i-1)*n+1:i*n] = f!(prob, st, y .+ z[(i-1)*n+1:i*n], t + RADAU_C[i] * h)
        end
        G = z .- h .* (kron(RADAU_A, Matrix{T}(I, n, n)) * F)
        Δ = M \ (-G)
        z .+= Δ
        nrm = sqrt(sum(abs2(Δ[k] / sc[(k-1)%n+1]) for k in eachindex(Δ)) / (3n))
        if it > 1
            θ = nrm / nrm_old
            θ >= 1 && return y, Inf, false  # diverging
            η = θ / (1 - θ)
        end
        nrm_old = nrm
        if (it > 1 && η * nrm <= κ) || nrm <= 1e-3 * κ
            c.iters = it
            converged = true
            break
        end
    end
    converged || return y, Inf, false
    z1, z2, z3 = z[1:n], z[n+1:2n], z[2n+1:3n]
    ynew = y .+ z3  # stiffly accurate: the last stage is the solution
    fixed && return ynew, 0.0, true
    f0 = f!(prob, st, y, t)
    raw = RADAU_E[1] .* z1 .+ RADAU_E[2] .* z2 .+ RADAU_E[3] .* z3 .- (h * RADAU_GAMMA0) .* f0
    err = lu(I - (h * RADAU_GAMMA0) .* c.J) \ raw
    st.nlu += 1
    return ynew, errnorm(err, y, ynew, rtol, atol), true
end
