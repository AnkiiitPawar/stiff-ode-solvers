# Dormand and Prince (1980): seven stages, a fifth-order solution and an
# embedded fourth-order one whose difference estimates the local error.

const DP_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
const DP_A = ((),
              (1 / 5,),
              (3 / 40, 9 / 40),
              (44 / 45, -56 / 15, 32 / 9),
              (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
              (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
              (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84))
const DP_B = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
const DP_BHAT = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)

init_cache(::DOPRI5, prob, y) = nothing

function step(::DOPRI5, cache, prob, st, y, t, h, rtol, atol, fresh)
    k = Vector{typeof(y)}(undef, 7)
    k[1] = f!(prob, st, y, t)
    for i in 2:7
        yi = copy(y)
        for j in 1:i-1
            DP_A[i][j] == 0 || (yi .+= (h * DP_A[i][j]) .* k[j])
        end
        k[i] = f!(prob, st, yi, t + DP_C[i] * h)
    end
    ynew = y .+ h .* sum(DP_B[i] .* k[i] for i in 1:6)
    err = h .* sum((DP_B[i] - DP_BHAT[i]) .* k[i] for i in 1:7)
    return ynew, errnorm(err, y, ynew, rtol, atol), true
end
