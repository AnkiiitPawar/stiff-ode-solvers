# Test problems with reference values.

"""
    robertson(; T=1e11)

Robertson's chemical kinetics (1966), the classic stiff test: rate constants
0.04, 1e4 and 3e7, y(0) = (1, 0, 0). `ROBERTSON_REF` is the solution at
t = 1e11 from the Bari Test Set for IVP Solvers (RADAU, rtol = 1.1e-18).
"""
robertson(; T=1e11) = ODEProblem(
    (y, t) -> [-0.04y[1] + 1e4y[2]y[3], 0.04y[1] - 1e4y[2]y[3] - 3e7y[2]^2, 3e7y[2]^2],
    [1.0, 0.0, 0.0], (0.0, T);
    jac=(y, t) -> [-0.04 1e4y[3] 1e4y[2]; 0.04 (-1e4y[3]-6e7y[2]) -1e4y[2]; 0.0 6e7y[2] 0.0])
const ROBERTSON_REF = [0.2083340149701255e-7, 0.8333360770334713e-13, 0.9999999791665050]

"""
    vanderpol(; ε=1e-6)

Van der Pol in the scaled form of the Bari Test Set: y1' = y2,
y2' = ((1 - y1^2) y2 - y1) / ε on [0, 2], y(0) = (2, 0). `VANDERPOL_REF` is
the Test Set's solution at t = 2 for ε = 1e-6.
"""
vanderpol(; ε=1e-6, T=2.0) = ODEProblem(
    (y, t) -> [y[2], ((1 - y[1]^2) * y[2] - y[1]) / ε],
    [2.0, 0.0], (0.0, T);
    jac=(y, t) -> [0.0 1.0; (-2y[1]*y[2] - 1)/ε (1 - y[1]^2)/ε])
const VANDERPOL_REF = [0.1706167732170483e1, -0.8928097010247975e0]

"""
    prothero_robinson(λ; T=1.0)

y' = λ (y - sin t) + cos t, y(0) = 0, with exact solution sin t for any λ.
λ ≪ 0 makes it stiff without changing the solution.
"""
prothero_robinson(λ; T=1.0) = ODEProblem((y, t) -> [λ * (y[1] - sin(t)) + cos(t)], [0.0], (0.0, T);
                                         jac=(y, t) -> fill(Float64(λ), 1, 1))

"""
    rotation()

y1' = -y2, y2' = y1 on [0, 1] from (1, 0): exact solution (cos t, sin t).
Smooth and non-stiff, for measuring orders of convergence.
"""
rotation() = ODEProblem((y, t) -> [-y[2], y[1]], [1.0, 0.0], (0.0, 1.0); jac=(y, t) -> [0.0 -1.0; 1.0 0.0])
