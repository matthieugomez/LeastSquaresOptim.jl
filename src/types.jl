 ##############################################################################
##
## Non Linear Least Squares
##
##############################################################################

type LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}
    x::Tx
    y::Ty
    f!::Tf
    J::TJ
    g!::Tg
    function LeastSquaresProblem(x, y, f!, J, g!)
        length(x) == size(J, 2) || throw(DimensionMismatch("x must have length size(J, 2)"))
        length(y) == size(J, 1) || throw(DimensionMismatch("y must have length size(J, 1)"))
        size(J, 1) >= size(J, 2) || throw(DimensionMismatch("size(J, 1) must be greater than size(J, 2)"))
        new(x, y, f!, J, g!)
    end
end
LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}(x::Tx, y::Ty, f!::Tf, J::TJ, g!::Tg) = LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}(x, y, f!, J, g!) 

typealias DenseLeastSquaresProblem{Tx, Ty, Tf, TJ<:StridedVecOrMat, Tg} LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}

typealias SparseLeastSquaresProblem{Tx, Ty, Tf, TJ<:SparseMatrixCSC, Tg} LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}

# Generate g! using ForwardDiff package
function LeastSquaresProblem(x::Vector, y::Vector, f!::Function, J::Matrix; chunk_size = 1)
    permf!(yp::Vector, xp::Vector) = f!(xp, yp)
    permg! = jacobian(permf!, mutates = true, chunk_size = chunk_size, output_length = length(y))
    g!(xp::Vector, Jp::Matrix) = permg!(Jp, xp)
    LeastSquaresProblem(x, y, f!, J, g!)
end

###############################################################################
##
## Non Linear Least Squares Allocated
## groups a LeastSquaresProblem with allocations
##
##############################################################################

# allocation for method
abstract AbstractMethod

# allocation for solver
abstract AbstractSolver


type LeastSquaresProblemAllocated{T <: LeastSquaresProblem, Tmethod <: AbstractMethod, Tsolver <: AbstractSolver}
     nls::T
     method::Tmethod
     solver::Tsolver
end

# Constructor
function LeastSquaresProblemAllocated{Tx, Ty, Tf, TJ, Tg}(
    nls::LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}; 
    method::Union{Void, Symbol} = nothing, solver::Union{Void, Symbol} = nothing)
    valsolver = default_solver(solver, TJ)
    valmethod = default_method(method, valsolver)
    LeastSquaresProblemAllocated(
        nls,
        AbstractMethod(nls, valmethod),
        AbstractSolver(nls, valmethod, valsolver))
end

## for dense matrices, default to cholesky ; otherwise iterative
function default_solver(x::Symbol, t::Type)
    if x ∉ (:qr, :cholesky, :iterative)
        throw("$x is not a valid solver. Choose between :qr, :cholesky, and :iterative.")
    end
    if x == :qr && t <: SparseMatrixCSC
        throw("solver = :qr is not available for sparse Jacobians. Choose between :cholesky and :iterative.")
    end
    Val{x}
end
default_solver{T<:StridedVecOrMat}(::Void, ::Type{T}) = Val{:qr}
default_solver(::Void, ::Type) = Val{:iterative}

## for iterative, default to levenberg_marquardt ; otherwise dogleg
function default_method(x::Symbol, ::Type)
    if x ∉ (:levenberg_marquardt, :dogleg)
        throw("$x is not a valid method. Choose between :levenberg_marquardt and :dogleg.")
    end
    Val{x}
end
default_method(::Void, ::Type{Val{:iterative}}) = Val{:levenberg_marquardt}
default_method(::Void, ::Type) = Val{:dogleg}


function LeastSquaresProblemAllocated(args...; kwargs...)
    LeastSquaresProblemAllocated(LeastSquaresProblem(args...); kwargs...)
end

# optimize
function optimize!(nls::LeastSquaresProblem; 
    method::Union{Void, Symbol} = nothing, 
    solver::Union{Void, Symbol} = nothing, 
    kwargs...)
    nlsp = LeastSquaresProblemAllocated(nls ; method = method, solver = solver)
    optimize!(nlsp; kwargs...)
end
###############################################################################
##
## Result of Non Linear Least Squares
##
##############################################################################

type LeastSquaresResult{Tx}
    method::ASCIIString
    minimizer::Tx
    ssr::Float64
    iterations::Int
    converged::Bool
    x_converged::Bool
    xtol::Real
    f_converged::Bool
    ftol::Real
    gr_converged::Bool
    grtol::Real
    f_calls::Int
    g_calls::Int
    mul_calls::Int
end

function LeastSquaresResult(method::ASCIIString, minimizer, ssr::Real, iterations::Int, converged::Bool, x_converged::Bool, xtol::Real, f_converged::Bool, ftol::Real, gr_converged::Bool, grtol::Real, f_calls::Int, g_calls::Int, mul_calls::Int)
    LeastSquaresResult(method, minimizer, convert(Float64, ssr), iterations, converged, x_converged, convert(Float64, xtol), f_converged, convert(Float64, ftol), gr_converged, convert(Float64, grtol), f_calls, g_calls, mul_calls)
end

function converged(r::LeastSquaresResult)
    return r.x_converged || r.f_converged || r.gr_converged
end


function Base.show(io::IO, r::LeastSquaresResult)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" r.method
    @printf io " * Minimizer: [%s]\n" join(r.minimizer, ",")
    @printf io " * Sum of squares at Minimum: %f\n" r.ssr
    @printf io " * Iterations: %d\n" r.iterations
    @printf io " * Convergence: %s\n" converged(r)
    @printf io " * |x - x'| < %.1e: %s\n" r.xtol r.x_converged
    @printf io " * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" r.ftol r.f_converged
    @printf io " * |g(x)| < %.1e: %s\n" r.grtol r.gr_converged
    @printf io " * Function Calls: %d\n" r.f_calls
    @printf io " * Gradient Calls: %d\n" r.g_calls
    @printf io " * Multiplication Calls: %d\n" r.mul_calls
    return
end

