using LeastSquaresOptim, LinearAlgebra

tests = ["nonlinearsolvers.jl", "nonlinearleastsquares.jl", "nonlinearfitting.jl", "bounds.jl"]

println("Running tests:")


for test in tests
	try
		include(test)
		println("\t\033[1m\033[32mPASSED\033[0m: $(test)")
	 catch e
	 	println("\t\033[1m\033[31mFAILED\033[0m: $(test)")
	 	showerror(stdout, e, backtrace())
	 	rethrow(e)
	 end
end

# test README
function rosenbrock(x)
	[1 - x[1], 100 * (x[2]-x[1]^2)]
end
x0 = zeros(2)
optimize(rosenbrock, x0, Dogleg())
optimize(rosenbrock, x0, LevenbergMarquardt())


function rosenbrock_f!(out, x)
 out[1] = 1 - x[1]
 out[2] = 100 * (x[2]-x[1]^2)
end
optimize!(LeastSquaresProblem(x = zeros(2), f! = rosenbrock_f!, output_length = 2, autodiff = :central), Dogleg())

# if you want to use gradient
function rosenbrock_g!(J, x)
    J[1, 1] = -1
    J[1, 2] = 0
    J[2, 1] = -200 * x[1]
    J[2, 2] = 100
end
optimize!(LeastSquaresProblem(x = zeros(2), f! = rosenbrock_f!, g! = rosenbrock_g!, output_length = 2), Dogleg())

# test scalar-valued function with multiple parameters (issue #41)
func(x) = sum(x.^2)
optimize(func, [1.0, 1.0], Dogleg())
optimize(func, [1.0, 1.0], LevenbergMarquardt())


# --- regression tests ---
using Test

# When J is supplied but y/output_length are omitted, output_length must default
# to the residual length size(J, 1), not size(J, 2) (only equal for square J).
let
    overdetermined!(out, x) = (out .= [x[1] - 1, x[2] - 2, x[3] - 3, x[1] + x[2], x[2] + x[3]])
    J = zeros(5, 3)
    p = LeastSquaresProblem(x = zeros(3), f! = overdetermined!, J = J)
    @test length(p.y) == 5
    r = optimize!(p, Dogleg())
    @test r.converged
end

# store_trace = true must populate a trace of OptimizationState
let
    r = optimize(rosenbrock, zeros(2), LevenbergMarquardt(); store_trace = true)
    @test eltype(r.tr.states) == LeastSquaresOptim.OptimizationState
    @test length(r.tr.states) >= 1
    r = optimize(rosenbrock, zeros(2), Dogleg(); store_trace = true)
    @test length(r.tr.states) >= 1
end