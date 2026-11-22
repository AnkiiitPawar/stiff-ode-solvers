using Test, StiffSolvers, LinearAlgebra

const S = StiffSolvers

@testset "StiffSolvers" begin
    @testset "orders of convergence with fixed steps" begin
        exact = [cos(1.0), sin(1.0)]
        for (m, p, ks) in ((DOPRI5(), 5, 2:5), (Rosenbrock23(), 2, 3:7), (RadauIIA5(), 5, 2:5), (BackwardEuler(), 1, 4:8))
            errs = [norm(solve(rotation(), m; dt=2.0^-k).u[end] - exact) for k in ks]
            observed = log2.(errs[1:end-1] ./ errs[2:end])
            @test all(abs.(observed .- p) .< 0.1)
        end
    end

    @testset "stability functions" begin
        # Exact forms: DOPRI5's degree-6 polynomial, the (2,3) Pade approximant
        # of Radau IIA, and 1/(1 - z) for backward Euler.
        for z in (-1.0, -3.0 + 2im, 0.5im, -0.3 - 0.7im)
            dp = sum(z^k / factorial(k) for k in 0:5) + z^6 / 600
            @test stability_function(DOPRI5(), z) ≈ dp rtol = 1e-13
            radau = (1 + 2z / 5 + z^2 / 20) / (1 - 3z / 5 + 3z^2 / 20 - z^3 / 60)
            @test stability_function(RadauIIA5(), z) ≈ radau rtol = 1e-12
            @test stability_function(BackwardEuler(), z) ≈ 1 / (1 - z) rtol = 1e-12
        end
        # A-stability: |R| <= 1 on the imaginary axis; L-stability: R -> 0 as z -> -infinity.
        for m in (Rosenbrock23(), RadauIIA5(), BackwardEuler())
            @test all(abs(stability_function(m, y * im)) <= 1 + 1e-12 for y in (0.01, 0.3, 1.0, 7.0, 100.0, 1e4))
            @test abs(stability_function(m, -1e8)) < 1e-6
        end
        # The explicit method's region is bounded: unstable on the negative axis beyond about -3.3.
        @test abs(stability_function(DOPRI5(), -3.0)) < 1
        @test abs(stability_function(DOPRI5(), -3.5)) > 1
    end

    @testset "Radau IIA embedded error estimate" begin
        # Derived here from the order-3 conditions; the same constants as Hairer's RADAU5.
        @test 1 / S.RADAU_GAMMA0 ≈ 3.637834252744496 rtol = 1e-14
        dd = [-(13 + 7sqrt(6)) / 3, (-13 + 7sqrt(6)) / 3, -1 / 3]
        @test S.RADAU_E ≈ -S.RADAU_GAMMA0 .* dd rtol = 1e-12
        @test S.RADAU_B ≈ [(16 - sqrt(6)) / 36, (16 + sqrt(6)) / 36, 1 / 9]
        @test sum(S.RADAU_A; dims=2)[:] ≈ S.RADAU_C  # row sums are the nodes
    end

    @testset "Robertson to t = 1e11 against the Test Set reference" begin
        sol = solve(robertson(), RadauIIA5(); rtol=1e-8, atol=1e-14, save_every=false)
        y = sol.u[end]
        @test sol.retcode == :success
        @test maximum(abs.(y .- ROBERTSON_REF) ./ ROBERTSON_REF) < 1e-6
        @test abs(sum(y) - 1) < 1e-13  # the linear invariant y1 + y2 + y3 = 1
        sol = solve(robertson(), Rosenbrock23(); rtol=1e-6, atol=1e-12, save_every=false)
        @test maximum(abs.(sol.u[end] .- ROBERTSON_REF) ./ ROBERTSON_REF) < 1e-3
        @test abs(sum(sol.u[end]) - 1) < 1e-13
    end

    @testset "Van der Pol (epsilon = 1e-6) against the Test Set reference" begin
        sol = solve(vanderpol(), RadauIIA5(); rtol=1e-8, atol=1e-8, save_every=false)
        @test maximum(abs.(sol.u[end] .- VANDERPOL_REF)) < 1e-7
        radau_steps = sol.stats.naccept
        sol = solve(vanderpol(), Rosenbrock23(); rtol=1e-6, atol=1e-6, save_every=false)
        @test maximum(abs.(sol.u[end] .- VANDERPOL_REF)) < 1e-3
        # The explicit method is held back by stability, not accuracy: over
        # the first tenth of the interval it already needs far more steps.
        short = vanderpol(T=0.2)
        explicit = solve(short, DOPRI5(); rtol=1e-6, atol=1e-6, save_every=false).stats.naccept
        implicit = solve(short, RadauIIA5(); rtol=1e-6, atol=1e-6, save_every=false).stats.naccept
        @test explicit > 100 * implicit
        @test radau_steps < 5000
    end

    @testset "tolerance proportionality" begin
        errs = [maximum(abs.(solve(vanderpol(), RadauIIA5(); rtol=tol, atol=tol, save_every=false).u[end] .- VANDERPOL_REF))
                for tol in (1e-4, 1e-6, 1e-8, 1e-10)]
        @test issorted(errs; rev=true)
        @test errs[end] < 1e-9
    end

    @testset "Prothero-Robinson: stiffness does not change the answer" begin
        for λ in (-1.0, -1e3, -1e6), m in (RadauIIA5(), Rosenbrock23())
            sol = solve(prothero_robinson(λ), m; rtol=1e-8, atol=1e-10, save_every=false)
            @test abs(sol.u[end][1] - sin(1.0)) < 1e-6
        end
    end

    @testset "finite-difference Jacobians" begin
        # Without an analytic Jacobian the implicit methods fall back to forward differences.
        p = robertson()
        q = ODEProblem(p.f, p.u0, (0.0, 1e5))
        b = solve(q, RadauIIA5(); rtol=1e-7, atol=1e-13, save_every=false).u[end]
        ref = solve(robertson(T=1e5), RadauIIA5(); rtol=1e-10, atol=1e-16, save_every=false).u[end]
        @test maximum(abs.(b .- ref) ./ abs.(ref)) < 1e-5
    end
end
