## LSMR works on any matrix with the following methods:
## eltype, size, A_mul_B!, Ac_mul_B!
## LSMR works on any vector with the following methods:
## eltype, length, scale!, norm

#############################################################################
## 
## Define preconditioned matrix
##
#############################################################################

struct PreconditionedMatrix{TA, Tp, Tx}
    A::TA
    preconditioner::Tp
    tmp::Tx
    tmp2::Tx
end
eltype(A::PreconditionedMatrix) = eltype(A.A)
size(A::PreconditionedMatrix, i::Integer) = size(A.A, i)
function A_mul_B!(α::Number, pm::PreconditionedMatrix{TA, Tp, Tx}, a::Tx, 
                β::Number, b) where {TA, Tp, Tx}
    A_ldiv_B!(a, pm.preconditioner, pm.tmp)
    A_mul_B!(α, pm.A, pm.tmp, β, b)
    return b
end
function Ac_mul_B!(α::Number, pm::PreconditionedMatrix{TA, Tp, Tx}, a, 
                β::Number, b::Tx) where {TA, Tp, Tx}
    T = eltype(b)
    β = convert(T, β)
    Ac_mul_B!(one(T), pm.A, a, zero(T), pm.tmp)
    A_ldiv_B!(pm.tmp, pm.preconditioner, pm.tmp2)
    if β != one(T)
        if β == zero(T)
            fill!(b, β)
        else
            scale!(b, β)
        end
    end
    axpy!(α, pm.tmp2, b)
    return b
end

#############################################################################
## 
## Define dampened matrix
##
#############################################################################


struct DampenedVector{Ty, Tx}
    y::Ty # dimension of f(x)
    x::Tx # dimension of x
end
eltype(a::DampenedVector) =  promote_type(eltype(a.y), eltype(a.x))
length(a::DampenedVector) = length(a.y) + length(a.x)
function scale!(a::DampenedVector, α::Number)
    scale!(a.y, α)
    scale!(a.x, α)
    return a
end
norm(a::DampenedVector) = sqrt(norm(a.y)^2 + norm(a.x)^2)

struct DampenedMatrix{TA, Tx}
    A::TA
    diagonal::Tx 
end
eltype(A::DampenedMatrix) = promote_type(eltype(A.A), eltype(A.diagonal))
function size(A::DampenedMatrix, dim::Integer)
    m, n = size(A.A)
    l = length(A.diagonal)
    dim == 1 ? (m + l) : 
    dim == 2 ? n : 1
end
function A_mul_B!(α::Number, mw::DampenedMatrix, a, 
                β::Number, b::DampenedVector)
    if β != 1
        scale!(b, β)
    end
    A_mul_B!(α, mw.A, a, 1, b.y)
    map!((z, x, y)-> z + α * x * y, b.x, b.x, a, mw.diagonal)
    return b
end
function Ac_mul_B!(α::Number, mw::DampenedMatrix, a::DampenedVector, 
                β::Number, b)
    T = eltype(b)
    β = convert(T, β)
    if β != one(T)
        if β == zero(T)
            fill!(b, β)
        else
            scale!(b, β)
        end
    end
    Ac_mul_B!(α, mw.A, a.y, one(T), b)
    map!((z, x, y)-> z + α * x * y, b, b, a.x, mw.diagonal)  
    return b
end

#############################################################################
## 
## Define inverse preconditioner
##
##############################################################################

struct InverseDiagonal{Tx}
    _::Tx
end
function A_ldiv_B!(x, ID::InverseDiagonal, y)
    map!(*, y, x, ID._)
end

#############################################################################
## 
## solve J'J \ J'y
##
## we use LSMR on Ax = y
## with A = J / sqrt(diag(J'J)) (diagonal preconditioner)
##
## LSMR works on any matrix with the following methods:
## eltype, size, A_mul_B!, Ac_mul_B!
##
##############################################################################
function getpreconditioner(nls::LeastSquaresProblem, optimizer::Dogleg, solver::LSMR{Void, Void})
    preconditioner! = function(x, J, out)
        colsumabs2!(out._, J)
        Tout = eltype(out._)
        map!(x -> x > zero(Tout) ? 1 / sqrt(x) : zero(Tout), out._, out._)
        return out
    end
    preconditioner = InverseDiagonal(_zeros(nls.x))
    return preconditioner!, preconditioner
end
function getpreconditioner(nls::LeastSquaresProblem, optimizer::Dogleg, solver::LSMR)
    return solver.preconditioner!, solver.preconditioner
end

struct LSMRAllocatedSolver{Tx0, Tx1, Tx2, Tx22, Tx3, Tx4, Tx5, Ty} <: AbstractAllocatedSolver
    preconditioner!::Tx0
    preconditioner::Tx1
    tmp::Tx2
    tmp2::Tx22
    v::Tx3
    h::Tx4
    hbar::Tx5
    u::Ty
end


function AbstractAllocatedSolver(nls::LeastSquaresProblem, optimizer::Dogleg, solver::LSMR)
    preconditioner!, preconditioner = getpreconditioner(nls, optimizer, solver)
    LSMRAllocatedSolver(preconditioner!, preconditioner, _zeros(nls.x), _zeros(nls.x), 
        _zeros(nls.x), _zeros(nls.x), _zeros(nls.x), _zeros(nls.y))
end

function A_ldiv_B!(x, J, y, A::LSMRAllocatedSolver)
    preconditioner!, preconditioner, tmp, tmp2, v, h, hbar, u = A.preconditioner!, A.preconditioner, A.tmp, A.tmp2, A.v, A.h, A.hbar, A.u

    # prepare x
    fill!(x, 0)

    # prepare b
    copy!(u, y)

    # prepare A
    fill!(tmp, 0)
    preconditioner!(x, J, preconditioner)
    A = PreconditionedMatrix(J, preconditioner, tmp, tmp2)

    # solve
    x, ch = lsmr!(x, A, u, v, h, hbar)
    A_ldiv_B!(x, preconditioner, tmp)
    copy!(x, tmp)
    return x, ch.mvps
end

##############################################################################
## 
## solve (J'J + diagm(damp)) \ J'y (used in LevenbergMarquardt)
## No need to solve exactly :
## "An Inexact Levenberg-Marquardt Method for Large Sparse Nonlinear Least Squares"
## Weight Holt (1985)
##
## We use LSMR on A x = b with
## A = |J          |  + diagonal preconditioner
##     |diag(damp) |
## b = vcat(y, zeros(damp))
##

##
##############################################################################
function getpreconditioner(nls::LeastSquaresProblem, optimizer::LevenbergMarquardt, ::LSMR{Void, Void})
    preconditioner! = function(x, J, damp, out)
        colsumabs2!(out._, J)
        Tout = eltype(out._)
        axpy!(one(Tout), damp, out._)
        map!(x -> x > zero(Tout) ? 1 / sqrt(x) : zero(Tout), out._, out._)
        return out
    end
    preconditioner = InverseDiagonal(_zeros(nls.x))
    return preconditioner!, preconditioner
end

function getpreconditioner(nls::LeastSquaresProblem, optimizer::LevenbergMarquardt, solver::LSMR)
    return solver.preconditioner!, solver.preconditioner
end

struct LSMRDampenedAllocatedSolver{Tx0, Tx1, Tx2, Tx22, Tx3, Tx4, Tx5, Tx6, Ty} <: AbstractAllocatedSolver
    preconditioner!::Tx0
    preconditioner::Tx1
    tmp::Tx2
    tmp2::Tx22
    v::Tx3
    h::Tx4
    hbar::Tx5
    zerosvector::Tx6
    u::Ty
end

function AbstractAllocatedSolver(nls::LeastSquaresProblem, optimizer::LevenbergMarquardt, solver::LSMR)
    preconditioner!, preconditioner = getpreconditioner(nls, optimizer, solver)
    LSMRDampenedAllocatedSolver(preconditioner!, preconditioner, _zeros(nls.x), _zeros(nls.x), _zeros(nls.x), _zeros(nls.x), _zeros(nls.x),  _zeros(nls.x), _zeros(nls.y))
end

function A_ldiv_B!(x, J, y, damp, A::LSMRDampenedAllocatedSolver)
    preconditioner!, preconditioner, tmp, tmp2, v, h, hbar, zerosvector, u = 
            A.preconditioner!, A.preconditioner, A.tmp, A.tmp2, A.v, A.h, A.hbar, A.zerosvector, A.u
    # prepare x
    fill!(x, 0)

    # prepare b
    copy!(u, y)
    fill!(zerosvector, 0)
    b = DampenedVector(u, zerosvector)

    # prepare A
    fill!(tmp, 0)
    preconditioner!(x, J, damp, preconditioner)
    map!(sqrt, damp, damp)
    A = PreconditionedMatrix(DampenedMatrix(J, damp), preconditioner, tmp, tmp2)
    # solve
    x, ch = lsmr!(x, A, b, v, h, hbar, btol = 0.5)
    A_ldiv_B!(x, preconditioner, tmp)
    copy!(x, tmp)
    return x, ch.mvps
end