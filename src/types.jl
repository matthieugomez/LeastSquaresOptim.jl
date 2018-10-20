 ##############################################################################
##
## Non Linear Least Squares
##
##############################################################################

struct LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}
    x::Tx
    y::Ty
    f!::Tf
    J::TJ
    g!::Tg
    function LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}(x, y, f!, J, g!) where {Tx, Ty, Tf, TJ, Tg}
        length(x) == size(J, 2) || throw(DimensionMismatch("x must have length size(J, 2)"))
        length(y) == size(J, 1) || throw(DimensionMismatch("y must have length size(J, 1)"))
        size(J, 1) >= size(J, 2) || throw(DimensionMismatch("size(J, 1) must be greater than size(J, 2)"))
        # test argument order
        g!(J, x)

        try 
            g!(J, x)
        catch
            throw("The order of argument and allocation arrays has been switched: use f!(fvec, x) and g!(J, x)")
        end
        # end test argument order
        new(x, y, f!, J, g!) 
    end
end


function LeastSquaresProblem(x::Tx, y::Ty, f!::Tf, J::TJ, g!::Tg) where {Tx, Ty, Tf, TJ, Tg}
    LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}(x, y, f!, J, g!)
end



function LeastSquaresProblem(;x = error("initial x required"), y = nothing, f! = error("initial f! required"), g! = nothing, J = nothing, output_length = 0, autodiff = :forward)
    if typeof(y) == Nothing
        if output_length == 0
            output_length = size(J, 2)
        end
        y = zeros(eltype(x), output_length)
    end
    if typeof(J) == Nothing
        J = zeros(eltype(x), length(y), length(x))
    end
    newg! = g!
    if typeof(g!) == Nothing
        # test argument order
        x0 = deepcopy(x)
        f!(y, x0)
        all(x0 .≈ x) || throw("The order of argument and allocation arrays has been switched: use f!(fvec, x)")
        # end test argument order
        @show autodiff
        if autodiff == :forward
            central_cache = DiffEqDiffTools.JacobianCache(similar(x), similar(y), similar(y))
            newg! = (Jp::Matrix, xp::Vector) -> DiffEqDiffTools.finite_difference_jacobian!(Jp, f, x, central_cache)
           end
        elseif autodiff == :central
            jac_cfg = ForwardDiff.JacobianConfig(f, y, x, ForwardDiff.Chunk(x))
            ForwardDiff.checktag(jac_cfg, f, x)
            y0 = deepcopy(y)
            newg! = (Jp::Matrix, xp::Vector) -> ForwardDiff.jacobian!(Jp, f!, y0, x, jac_cfg, Val{False}())
        end

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
abstract type AbstractOptimizer end
struct Dogleg <: AbstractOptimizer end
struct LevenbergMarquardt <: AbstractOptimizer end



# solver
abstract type AbstractSolver end
struct QR <: AbstractSolver end
struct Cholesky <: AbstractSolver end
struct LSMR{T1, T2} <: AbstractSolver
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
default_solver(::Nothing, J::StridedVecOrMat) = QR()
default_solver(::Nothing, J) = LSMR()

## for LSMR, default to levenberg_marquardt ; otherwise dogleg
default_optimizer(x::AbstractOptimizer, y) = x
default_optimizer(::Nothing, ::LSMR) = LevenbergMarquardt()
default_optimizer(::Nothing, ::AbstractSolver) = Dogleg()



abstract type AbstractAllocatedOptimizer end
abstract type AbstractAllocatedSolver end

mutable struct LeastSquaresProblemAllocated{Tx, Ty, Tf, TJ, Tg, Toptimizer <: AbstractAllocatedOptimizer, Tsolver <: AbstractAllocatedSolver}
    x::Tx
    y::Ty
    f!::Tf
    J::TJ
    g!::Tg
    optimizer::Toptimizer
    solver::Tsolver
end

# Constructor
function LeastSquaresProblemAllocated(nls::LeastSquaresProblem, optimizer::Union{Nothing, AbstractOptimizer}, solver::Union{Nothing, AbstractSolver})
    solver = default_solver(solver, nls.J)
    optimizer = default_optimizer(optimizer, solver)
    LeastSquaresProblemAllocated(
        nls.x, nls.y, nls.f!, nls.J, nls.g!, AbstractAllocatedOptimizer(nls, optimizer), AbstractAllocatedSolver(nls, optimizer, solver))
end
function LeastSquaresProblemAllocated(args...; kwargs...)
    LeastSquaresProblemAllocated(LeastSquaresProblem(args...); kwargs...)
end

# optimize
function optimize!(nls::LeastSquaresProblem, optimizer::Union{Nothing, AbstractOptimizer} = nothing, solver::Union{Nothing, AbstractSolver} = nothing; autodiff = :forward, kwargs...)
    nlsp = LeastSquaresProblemAllocated(nls, optimizer, solver, autodiff = autodiff)
    optimize!(nlsp; kwargs...)
end

###############################################################################
##
## Optim-like syntax
##
##############################################################################

function optimize(f, x, t::AbstractOptimizer; autodiff = :forward, kwargs...)
    optimize!(LeastSquaresProblem(x = deepcopy(x), f! = (out, x) -> copyto!(out, f(x)), output_length = length(f(x)), autodiff = autodiff), t; kwargs...)
end

###############################################################################
##
## Result of Non Linear Least Squares
##
##############################################################################

struct LeastSquaresResult{Tx}
    optimizer::String
    minimizer::Tx
    ssr::Float64
    iterations::Int
    converged::Bool
    x_converged::Bool
    x_tol::Real
    f_converged::Bool
    f_tol::Real
    g_converged::Bool
    g_tol::Real
    tr::OptimizationTrace
    f_calls::Int
    g_calls::Int
    mul_calls::Int
end

function LeastSquaresResult(optimizer::String, minimizer, ssr::Real, iterations::Int, converged::Bool, x_converged::Bool, x_tol::Real, f_converged::Bool, f_tol::Real, g_converged::Bool, g_tol::Real, tr::OptimizationTrace, f_calls::Int, g_calls::Int, mul_calls::Int)
    LeastSquaresResult(optimizer, minimizer, convert(Float64, ssr), iterations, converged, x_converged, convert(Float64, x_tol), f_converged, convert(Float64, f_tol), g_converged, convert(Float64, g_tol), tr, f_calls, g_calls, mul_calls)
end

function converged(r::LeastSquaresResult)
    return r.x_converged || r.f_converged || r.g_converged
end


function Base.show(io::IO, r::LeastSquaresResult)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" r.optimizer
    @printf io " * Minimizer: [%s]\n" join(r.minimizer, ",")
    @printf io " * Sum of squares at Minimum: %f\n" r.ssr
    @printf io " * Iterations: %d\n" r.iterations
    @printf io " * Convergence: %s\n" converged(r)
    @printf io " * |x - x'| < %.1e: %s\n" r.x_tol r.x_converged
    @printf io " * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" r.f_tol r.f_converged
    @printf io " * |g(x)| < %.1e: %s\n" r.g_tol r.g_converged
    @printf io " * Function Calls: %d\n" r.f_calls
    @printf io " * Gradient Calls: %d\n" r.g_calls
    @printf io " * Multiplication Calls: %d\n" r.mul_calls
    return
end

