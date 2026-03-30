##############################################################################
##
## Non Linear Least Squares Problem
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
        new(x, y, f!, J, g!) 
    end
end

function LeastSquaresProblem(x::Tx, y::Ty, f!::Tf, J::TJ, g!::Tg) where {Tx, Ty, Tf, TJ, Tg}
    LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}(x, y, f!, J, g!)
end

"""
    LeastSquaresProblem(; x, f!, output_length, y, g!, J, autodiff)

Define a non-linear least squares problem.

## Required keyword arguments
- `x`: Initial vector of parameters.
- `f!`: In-place residual function `f!(out, x)` that writes `f(x)` into `out`.

## Optional keyword arguments
- `output_length::Int = 0`: Length of the residual vector. Required if neither `y` nor `J` is provided.
- `y = nothing`: Pre-allocated residual vector. If `nothing`, allocated automatically using `output_length`.
- `g! = nothing`: Jacobian function `g!(J, x)` that writes the Jacobian at `x` into `J`. If `nothing`, computed via automatic differentiation (see `autodiff`).
- `J = nothing`: Pre-allocated Jacobian matrix. If `nothing`, allocated as a dense `Matrix`.
- `autodiff::Symbol = :central`: Method for automatic Jacobian computation when `g!` is not provided. `:central` for central finite differences, `:forward` for `ForwardDiff.jl`.
"""
function LeastSquaresProblem(;x = error("initial x required"), y = nothing, f! = error("initial f! required"), g! = nothing, J = nothing, output_length = 0, autodiff = :central)
    if isnothing(y)
        if output_length == 0
            if isnothing(J)
                error("specify J or output_length")
            else
                output_length = size(J, 2)
            end
        end
        y = zeros(eltype(x), output_length)
    end
    if isnothing(J)
        J = zeros(eltype(x), length(y), length(x))
    end
    newg! = g!
    if isnothing(g!)
        if autodiff == :central
            central_cache = JacobianCache(similar(x), similar(y), similar(y))
            newg! = (J::Matrix, xp::Vector) -> finite_difference_jacobian!(J, f!, xp, central_cache)
        elseif autodiff == :forward
            jac_cfg = JacobianConfig(f!, y, x, Chunk(x))
            checktag(jac_cfg, f!, x)
            newg! = (J::Matrix, xp::Vector) -> jacobian!(J, f!, deepcopy(y), xp, jac_cfg, Val{false}())
        else
            throw(DomainError(autodiff, "Invalid automatic differentiation method."))
        end
    end
    LeastSquaresProblem(x, y , f!, J, newg!)
end


###############################################################################
##
## Optimizer and SOlver
##
##############################################################################


# solver
abstract type AbstractSolver end
struct QR <: AbstractSolver end
struct Cholesky <: AbstractSolver end
struct LSMR{T1, T2} <: AbstractSolver
    preconditioner!::T1
    P::T2
end
LSMR() = LSMR(nothing, nothing)


# optimizer
abstract type AbstractOptimizer{T} end
struct Dogleg{T} <: AbstractOptimizer{T}
    solver::T
end
Dogleg() = Dogleg(nothing)
struct LevenbergMarquardt{T} <: AbstractOptimizer{T}
    solver::T
end
LevenbergMarquardt() = LevenbergMarquardt(nothing)


_solver(x::AbstractOptimizer) = x.solver
_solver(x::Nothing) = nothing



## Shared constants for optimizers
const MIN_Δ = 1e-16 # minimum trust region radius
const MAX_Δ = 1e16 # maximum trust region radius
const MIN_STEP_QUALITY = 1e-3
const MIN_DIAGONAL = 1e-6
const MAX_DIAGONAL = 1e32

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
default_optimizer(x::Dogleg, y::AbstractSolver) = Dogleg(y)
default_optimizer(x::LevenbergMarquardt, y::AbstractSolver) = LevenbergMarquardt(y)
default_optimizer(::Nothing, ::LSMR) = LevenbergMarquardt(LSMR())
default_optimizer(::Nothing, x) = Dogleg(x)




##############################################################################
##
## Non Linear Least Squares Problem Allocated
##
##############################################################################

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
function LeastSquaresProblemAllocated(nls::LeastSquaresProblem, optimizer::Union{Nothing, AbstractOptimizer})
    solver = default_solver(_solver(optimizer), nls.J)
    optimizer = default_optimizer(optimizer, solver)
    LeastSquaresProblemAllocated(
        nls.x, nls.y, nls.f!, nls.J, nls.g!, AbstractAllocatedOptimizer(nls, optimizer), AbstractAllocatedSolver(nls, optimizer))
end
function LeastSquaresProblemAllocated(args...; kwargs...)
    LeastSquaresProblemAllocated(LeastSquaresProblem(args...); kwargs...)
end
###############################################################################
##
## Optim-like syntax
##
##############################################################################

"""
    optimize(f, x, optimizer; autodiff = :central, kwargs...)

Minimize `sum(f(x).^2)` with respect to `x`.

## Arguments
- `f`: Function from `x` to a residual vector.
- `x`: Initial vector of parameters (will be copied).
- `optimizer`: `Dogleg()` or `LevenbergMarquardt()`, optionally wrapping a solver (e.g. `Dogleg(LeastSquaresOptim.QR())`).

## Keyword arguments
- `autodiff::Symbol = :central`: `:central` for central finite differences, `:forward` for `ForwardDiff.jl`.

All other keyword arguments are forwarded to [`optimize!`](@ref) (see below for the full list of convergence and display options).
"""
function optimize(f, x, t::AbstractOptimizer; autodiff = :central, kwargs...)
    optimize!(LeastSquaresProblem(x = deepcopy(x), f! = (out, x) -> copyto!(out, f(x)), output_length = length(f(x)), autodiff = autodiff), t; kwargs...)
end

"""
    optimize!(nls::LeastSquaresProblem, optimizer = nothing; kwargs...)

Solve the least squares problem `nls` in place, modifying `nls.x`.

## Arguments
- `nls`: A [`LeastSquaresProblem`](@ref).
- `optimizer`: `Dogleg()`, `LevenbergMarquardt()`, or `nothing` (default chosen based on Jacobian type: `Dogleg(QR())` for dense, `LevenbergMarquardt(LSMR())` for sparse).

## Keyword arguments
- `x_tol::Number = 1e-8`: Convergence tolerance on parameter changes: `maximum(abs, δx) ≤ x_tol`.
- `f_tol::Number = 1e-8`: Convergence tolerance on objective change: `|f(x) - f(x')| / |f(x)| ≤ f_tol`.
- `g_tol::Number = 1e-8`: Convergence tolerance on gradient: `maximum(abs, J'f) ≤ g_tol`.
- `iterations::Integer = 1_000`: Maximum number of iterations.
- `Δ::Number`: Initial trust region radius. Default is `1.0` for `Dogleg` and `10.0` for `LevenbergMarquardt`.
- `lower::Vector = eltype(x)[]`: Lower bounds on parameters (empty = no bounds).
- `upper::Vector = eltype(x)[]`: Upper bounds on parameters (empty = no bounds).
- `store_trace::Bool = false`: Store the optimization trace.
- `show_trace::Bool = false`: Print the optimization trace during iteration.
- `show_every::Int = 1`: Print trace every `show_every` iterations (only used when `show_trace = true`).
"""
function optimize!(nls::LeastSquaresProblem, optimizer::Union{Nothing, AbstractOptimizer} = nothing; kwargs...)
    optimize!(LeastSquaresProblemAllocated(nls, optimizer); kwargs...)
end




###############################################################################
##
## Result of Non Linear Least Squares
##
##############################################################################

struct LeastSquaresResult{Tx, TJ}
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
    jacobian::TJ
end

function LeastSquaresResult(optimizer::String, minimizer, ssr::Real, iterations::Int, converged::Bool, x_converged::Bool, x_tol::Real, f_converged::Bool, f_tol::Real, g_converged::Bool, g_tol::Real, tr::OptimizationTrace, f_calls::Int, g_calls::Int, mul_calls::Int, jacobian)
    LeastSquaresResult(optimizer, minimizer, convert(Float64, ssr), iterations, converged, x_converged, convert(Float64, x_tol), f_converged, convert(Float64, f_tol), g_converged, convert(Float64, g_tol), tr, f_calls, g_calls, mul_calls, jacobian)
end

function converged(r::LeastSquaresResult)
    return r.x_converged || r.f_converged || r.g_converged
end


function Base.show(io::IO, r::LeastSquaresResult)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Status: %s\n" (converged(r) ? "success" : "failure (reached maximum number of iterations)")
    @printf io "\n"
    @printf io " * Candidate solution\n"
    @printf io "    Final objective value:     %.6e\n" r.ssr
    @printf io "\n"
    @printf io " * Found with\n"
    @printf io "    Algorithm:     %s\n" r.optimizer
    @printf io "\n"
    @printf io " * Convergence measures\n"
    @printf io "    |x - x'|               %s %.1e\n" (r.x_converged ? "≤" : "≰") r.x_tol
    @printf io "    |f(x) - f(x')| / |f(x)| %s %.1e\n" (r.f_converged ? "≤" : "≰") r.f_tol
    @printf io "    |g(x)|                 %s %.1e\n" (r.g_converged ? "≤" : "≰") r.g_tol
    @printf io "\n"
    @printf io " * Work counters\n"
    @printf io "    Iterations:    %d\n" r.iterations
    @printf io "    f(x) calls:    %d\n" r.f_calls
    @printf io "    J(x) calls:    %d\n" r.g_calls
    @printf io "    mul! calls:    %d\n" r.mul_calls
    return
end
