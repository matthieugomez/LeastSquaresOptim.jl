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

function LeastSquaresProblem(;x = error("initial x required"), y = nothing, f! = error("initial f! required"), g! = nothing, J = nothing, output_length = 0, chunk_size = 1)
    if typeof(y) == Void
        if output_length == 0
            output_length = size(J, 2)
        end
        y = zeros(eltype(x), output_length)
    end
    if typeof(J) == Void
        J = zeros(eltype(x), length(y), length(x))
    end
    newg! = g!
    if typeof(g!) == Void
        permf!(yp::Vector, xp::Vector) = f!(xp, yp)
        y0 = deepcopy(y)
        newg! = (xp::Vector, Jp::Matrix) -> ForwardDiff.jacobian!(Jp, permf!, y0, x,  ForwardDiff.JacobianConfig{chunk_size}(x))
    end
    LeastSquaresProblem(x, y , f!, J, newg!)
end





###############################################################################
##
## Non Linear Least Squares Allocated
## groups a LeastSquaresProblem with allocations
##
##############################################################################
# optimizer
abstract AbstractOptimizer
immutable Dogleg <: AbstractOptimizer end
immutable LevenbergMarquardt <: AbstractOptimizer end


# solver
abstract AbstractSolver
immutable QR <: AbstractSolver end
immutable Cholesky <: AbstractSolver end
type LSMR{T1, T2} <: AbstractSolver
    preconditioner!::T1
    preconditioner::T2
end
LSMR() = LSMR(nothing, nothing)

## for dense matrices, default to cholesky ; otherwise LSMR
function default_solver(x::AbstractSolver, J)
    if (typeof(x) <: QR) && (typeof(J) <: SparseMatrixCSC)
        throw("solver QR() is not available for sparse Jacobians. Choose between Cholesky() and LSMR()")
    end
    x
end
default_solver(::Void, J::StridedVecOrMat) = QR()
default_solver(::Void, J) = LSMR()

## for LSMR, default to levenberg_marquardt ; otherwise dogleg
default_optimizer(x::AbstractOptimizer, y) = x
default_optimizer(::Void, ::LSMR) = LevenbergMarquardt()
default_optimizer(::Void, ::AbstractSolver) = Dogleg()



abstract AbstractAllocatedOptimizer
abstract AbstractAllocatedSolver

type LeastSquaresProblemAllocated{Tx, Ty, Tf, TJ, Tg, Toptimizer <: AbstractAllocatedOptimizer, Tsolver <: AbstractAllocatedSolver}
    x::Tx
    y::Ty
    f!::Tf
    J::TJ
    g!::Tg
    optimizer::Toptimizer
    solver::Tsolver
end

# Constructor
function LeastSquaresProblemAllocated(nls::LeastSquaresProblem, optimizer::Union{Void, AbstractOptimizer}, solver::Union{Void, AbstractSolver})
    solver = default_solver(solver, nls.J)
    optimizer = default_optimizer(optimizer, solver)
    LeastSquaresProblemAllocated(
        nls.x, nls.y, nls.f!, nls.J, nls.g!, AbstractAllocatedOptimizer(nls, optimizer), AbstractAllocatedSolver(nls, optimizer, solver))
end
function LeastSquaresProblemAllocated(args...; kwargs...)
    LeastSquaresProblemAllocated(LeastSquaresProblem(args...); kwargs...)
end

# optimize
function optimize!(nls::LeastSquaresProblem, optimizer::Union{Void, AbstractOptimizer} = nothing, solver::Union{Void, AbstractSolver} = nothing; kwargs...)
    nlsp = LeastSquaresProblemAllocated(nls, optimizer, solver)
    optimize!(nlsp; kwargs...)
end


###############################################################################
##
## Result of Non Linear Least Squares
##
##############################################################################

type LeastSquaresResult{Tx}
    optimizer::String
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
    tr::OptimizationTrace
    f_calls::Int
    g_calls::Int
    mul_calls::Int
end

function LeastSquaresResult(optimizer::String, minimizer, ssr::Real, iterations::Int, converged::Bool, x_converged::Bool, xtol::Real, f_converged::Bool, ftol::Real, gr_converged::Bool, grtol::Real, tr::OptimizationTrace, f_calls::Int, g_calls::Int, mul_calls::Int)
    LeastSquaresResult(optimizer, minimizer, convert(Float64, ssr), iterations, converged, x_converged, convert(Float64, xtol), f_converged, convert(Float64, ftol), gr_converged, convert(Float64, grtol), tr, f_calls, g_calls, mul_calls)
end

function converged(r::LeastSquaresResult)
    return r.x_converged || r.f_converged || r.gr_converged
end


function Base.show(io::IO, r::LeastSquaresResult)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" r.optimizer
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

