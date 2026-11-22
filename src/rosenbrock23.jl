# The Rosenbrock pair of MATLAB's ode23s (Shampine and Reichelt 1997). Each
# step solves three linear systems with one matrix W = I - h d J and no Newton
# iteration; the second-order solution is L-stable and the error estimate is
# third order.

const ROS_D = 1 / (2 + sqrt(2))
const ROS_E32 = 6 + sqrt(2)

mutable struct RosCache{M,V}
    J::M
    T::V  # partial derivative of f with respect to t
end

init_cache(::Rosenbrock23, prob, y) = RosCache(zeros(eltype(y), length(y), length(y)), zero(y))

function step(::Rosenbrock23, c::RosCache, prob, st, y, t, h, rtol, atol, fresh)
    F0 = f!(prob, st, y, t)
    if fresh
        c.J = jacobian(prob, st, y, t)
        δ = sqrt(eps(Float64)) * max(abs(t), 1.0)
        c.T = (f!(prob, st, y, t + δ) .- F0) ./ δ
    end
    W = lu(I - (h * ROS_D) .* c.J)
    st.nlu += 1
    k1 = W \ (F0 .+ (h * ROS_D) .* c.T)
    F1 = f!(prob, st, y .+ (h / 2) .* k1, t + h / 2)
    k2 = (W \ (F1 .- k1)) .+ k1
    ynew = y .+ h .* k2
    F2 = f!(prob, st, ynew, t + h)
    k3 = W \ (F2 .- ROS_E32 .* (k2 .- F1) .- 2 .* (k1 .- F0) .+ (h * ROS_D) .* c.T)
    err = (h / 6) .* (k1 .- 2 .* k2 .+ k3)
    return ynew, errnorm(err, y, ynew, rtol, atol), true
end
