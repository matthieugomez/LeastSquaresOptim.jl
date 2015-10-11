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

