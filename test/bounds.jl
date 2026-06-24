using LeastSquaresOptim, LinearAlgebra, Test

function rosenbrock(x)
    [1 - x[1], 100 * (x[2]-x[1]^2)]
end

@testset "bounds" begin
    for opt in (Dogleg(), LevenbergMarquardt())
        # Bound inactive at the optimum: rosenbrock's minimum (1, 1) is interior to
        # x ≥ 0, so the solver must reach it while staying feasible.
        r = optimize(rosenbrock, zeros(2), opt, lower = [0.0, 0.0])
        @test r.converged
        @test all(r.minimizer .>= -1e-8)
        @test norm(r.minimizer - [1.0, 1.0]) <= 1e-6

        # Lower bound active at the optimum (improvement #1). x₁ wants 0.5 but is
        # held at its bound 1; x₂ is free and wants 3. The raw gradient there is
        # (0.5, 0), so with x_tol/f_tol disabled only the *projected* gradient can
        # certify convergence — g_converged must fire at (1, 3).
        flo!(out, x) = (out[1] = x[1] - 0.5; out[2] = x[2]^2 - 9)
        p = LeastSquaresProblem(x = [2.0, 1.0], f! = flo!, output_length = 2)
        r = optimize!(p, opt, lower = [1.0, -100.0], x_tol = 1e-50, f_tol = 1e-50)
        @test r.converged
        @test r.g_converged
        @test r.minimizer[1] >= 1.0 - 1e-8
        @test norm(r.minimizer - [1.0, 3.0]) <= 1e-6

        # Upper bound active at the optimum: x₁ wants 5 but is held at its bound 2;
        # the gradient (-3, 0) points out of the box.
        fhi!(out, x) = (out[1] = x[1] - 5; out[2] = x[2]^2 - 4)
        p = LeastSquaresProblem(x = [0.0, 1.0], f! = fhi!, output_length = 2)
        r = optimize!(p, opt, upper = [2.0, 100.0], x_tol = 1e-50, f_tol = 1e-50)
        @test r.converged
        @test r.g_converged
        @test r.minimizer[1] <= 2.0 + 1e-8
        @test norm(r.minimizer - [2.0, 2.0]) <= 1e-6
    end
end
