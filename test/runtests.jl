using LeastSquaresOptim

tests = ["nonlinearsolvers.jl", "nonlinearleastsquares.jl", "nonlinearfitting.jl"]

println("Running tests:")

# to pass dense lsmr
import LinearAlgebra: mul!
function mul!(C::Vector{Float64}, A::Matrix{Float64}, B::Vector{Float64}, α::Number, β::Number)
    gemm!('N', 'N', convert(Float64, α), A, B, convert(Float64, β), C)
end

function mul!(C::Vector{Float64}, A::Adjoint{Float64, Matrix{Float64}}, B::Vector{Float64}, α::Number, β::Number)
    gemm!('C', 'N', convert(Float64, α), A', B, convert(Float64, β), C)
end

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

