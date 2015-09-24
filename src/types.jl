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
end

typealias DenseLeastSquaresProblem{Tx, Ty, Tf, TJ<:StridedVecOrMat, Tg} LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}

typealias SparseLeastSquaresProblem{Tx, Ty, Tf, TJ<:SparseMatrixCSC, Tg} LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}

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

type LeastSquaresProblemAllocated{T <: LeastSquaresProblem, Tmethod <: AbstractMethod, Tsolve <: AbstractSolver}
    nls::T
    method::Tmethod
    solve::Tsolve
end

function LeastSquaresProblemAllocated{Tx, Ty, Tf, TJ, Tg}(
    nls::LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}; 
    method::Union{Void, Symbol} = nothing, solver::Union{Void, Symbol} = nothing)
    valsolver = default_solver(solver, TJ)
    valmethod = default_method(method, valsolver)
    LeastSquaresProblemAllocated(nls,
    allocate(nls, valmethod), 
    allocate(nls, valmethod, valsolver))
end

# or dense matrices, default to factorization f, otherwise iterative
default_solver(x::Symbol, ::Type) = Val{x}
default_solver{T<:StridedVecOrMat}(::Void, ::Type{T}) = Val{:factorization}
default_solver(::Void, ::Type) = Val{:iterative}

# for iterative, default to levenberg_marquardt ; otherwise dogleg
default_method(x::Symbol, ::Type) = Val{x}
default_method(::Void, ::Type{Val{:iterative}}) = Val{:levenberg_marquardt}
default_method(::Void, ::Type) = Val{:dogleg}

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

type LeastSquaresResult
    method::Symbol
    x::Any
    ssr::Real
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

