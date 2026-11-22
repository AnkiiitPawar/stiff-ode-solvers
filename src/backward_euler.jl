# Backward Euler with fixed steps: y1 = y0 + h f(y1, t + h), solved by Newton
# iteration with the Jacobian taken at y0. Order 1, L-stable.

init_cache(::BackwardEuler, prob, y) = nothing

function step(::BackwardEuler, cache, prob, st, y, t, h, rtol, atol, fresh)
    J = jacobian(prob, st, y, t + h)
    M = lu(I - h .* J)
    st.nlu += 1
    y1 = copy(y)
    for _ in 1:50
        st.nnewton += 1
        Δ = M \ (y .+ h .* f!(prob, st, y1, t + h) .- y1)
        y1 = y1 .+ Δ
        norm(Δ) <= 1e-14 * (1 + norm(y1)) && return y1, 0.0, true
    end
    return y1, 0.0, false
end
