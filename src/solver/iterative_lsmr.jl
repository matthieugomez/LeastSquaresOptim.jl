##############################################################################
## 
## LSMR with diagonal preconditioner, ie A -> A / sqrt(A'A)
##
##############################################################################

type PMatrix{TA, Tx}
    A::TA
    normalization::Tx 
    tmp::Tx
end
eltype(A::PMatrix) = eltype(A.A)
size(A::PMatrix, i::Integer) = size(A.A, i)

function A_mul_B!{TA, Tx}(α::Number, pm::PMatrix{TA, Tx}, a::Tx, 
                β::Number, b)
    map!(*, pm.tmp, a, pm.normalization)
    A_mul_B!(α, pm.A, pm.tmp, β, b)
    return b
end

function Ac_mul_B!{TA, Tx}(α::Number, pm::PMatrix{TA, Tx}, a, 
                β::Number, b::Tx)
    Ac_mul_B!(1, pm.A, a, 0, pm.tmp)
    map!(*, pm.tmp, pm.tmp, pm.normalization)
    if β != 1
        if β == 0
            fill!(b, 0)
        else
            scale!(b, β)
        end
    end
    axpy!(α, pm.tmp, b)
    return b
end

type PreconditionedMatrix{TA, Tx}
    A::TA
    normalization::Tx  # 1 / sqrt(diag(A'A))
    tmp::Tx # a storage vector of size(A, 2)
end

function lsmr!(x, A::PreconditionedMatrix, r, v, h, hbar; kwargs...)
    PA = PMatrix(A.A, A.normalization, A.tmp)
    result = lsmr!(x, PA, r, v, h, hbar; kwargs...)
    map!(*, x, x, A.normalization)
    return result
end

##############################################################################
## 
## LSMR with matrix A = |J         |
##                      |diag(dtd) |
##
##############################################################################

type DampenedVector{Ty, Tx}
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

type DampenedMatrix{TA, Tx}
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

function A_mul_B!{TA, Tx, Ty}(α::Number, mw::DampenedMatrix{TA, Tx}, a::Tx, 
                β::Number, b::DampenedVector{Ty, Tx})
    if β != 1
        if β == 0
            fill!(b, 0)
        else
            scale!(b, β)
        end
    end
    A_mul_B!(α, mw.A, a, 1, b.y)
    map!((z, x, y)-> z + α * x * y, b.x, b.x, a, mw.diagonal)
    return b
end

function Ac_mul_B!{TA, Tx, Ty}(α::Number, mw::DampenedMatrix{TA, Tx}, a::DampenedVector{Ty, Tx}, 
                β::Number, b::Tx)
    if β != 1
        if β == 0
            fill!(b, 0)
        else
            scale!(b, β)
        end
    end
    Ac_mul_B!(α, mw.A, a.y, 1, b)
    map!((z, x, y)-> z + α * x * y, b, b, a.x, mw.diagonal)  
    return b
end

##############################################################################
## 
## Dogleg : solve J'J \ J'y
##
## we use LSMR for the problem J'J \ J' fcur 
## with 1/sqrt(diag(J'J)) as preconditioner
##
##############################################################################

type DoglegLSMR{Tx1, Tx2, Tx3, Tx4, Tx5, Tx6, Ty} <: AbstractSolver
    normalization::Tx1
    tmp::Tx2
    v::Tx3
    h::Tx4
    hbar::Tx5
    zerosvector::Tx6
    b::Ty
end

function allocate(nls::LeastSquaresProblem,
    ::Type{Val{:dogleg}}, ::Type{Val{:iterative}})
    DoglegLSMR(_zeros(nls.x), _zeros(nls.x), _zeros(nls.x), 
        _zeros(nls.x), _zeros(nls.x),  _zeros(nls.x), _zeros(nls.y))
end

function solve!{T, Tmethod <: Dogleg, Tsolve <: DoglegLSMR}(
    anls::LeastSquaresProblemAllocated{T, Tmethod, Tsolve})
    normalization, tmp, v, h, hbar, b = anls.solve.normalization, anls.solve.tmp, anls.solve.v, anls.solve.h, anls.solve.hbar, anls.solve.b
    J, y = anls.nls.J, anls.nls.y
    δgn = anls.method.δgn

    # prepare x
    fill!(δgn, 0)

    # prepare b
    copy!(b, y)

    # prepare A
    colsumabs2!(normalization, J)
    map!(x -> x > 0 ? 1 / sqrt(x) : 0, normalization, normalization)
    A = PreconditionedMatrix(J, normalization, tmp)

    # solve
    x, ch = lsmr!(δgn, A, b, v, h, hbar)
    return ch.mvps
end

##############################################################################
## 
## LevenbergMarquardt: solve (J'J + λ dtd) \ J'y
## See "An Inexact Levenberg-Marquardt Method for Large Sparse Nonlinear Least Squares"
## Weight Holt (1985)
##
##############################################################################

type LevenbergMarquardtLSMR{Tx1, Tx2, Tx3, Tx4, Tx5, Tx6, Ty} <: AbstractSolver
    normalization::Tx1
    tmp::Tx2
    v::Tx3
    h::Tx4
    hbar::Tx5
    zerosvector::Tx6
    u::Ty
end

function allocate(nls::LeastSquaresProblem,
    ::Type{Val{:levenberg_marquardt}}, ::Type{Val{:iterative}})
    LevenbergMarquardtLSMR(_zeros(nls.x), _zeros(nls.x), 
        _zeros(nls.x), _zeros(nls.x), _zeros(nls.x), _zeros(nls.x), _zeros(nls.y))
end

function solve!{T, Tmethod <: LevenbergMarquardt, Tsolve <: LevenbergMarquardtLSMR}(
    anls::LeastSquaresProblemAllocated{T, Tmethod, Tsolve}, λ)
    normalization, tmp, v, h, hbar, zerosvector, u = anls.solve.normalization, anls.solve.tmp, anls.solve.v, anls.solve.h, anls.solve.hbar, anls.solve.zerosvector, anls.solve.u
    δx, dtd = anls.method.δx, anls.method.dtd
    y, J = anls.nls.y, anls.nls.J

    # prepare x
    fill!(δx, 0)

    # prepare b
    copy!(u, y)
    fill!(zerosvector, 0)
    b = DampenedVector(u, zerosvector)

    # prepare A
    fill!(tmp, 0)
    copy!(normalization, dtd)
    map!(x -> max(x, MIN_DIAGONAL), dtd, dtd)
    scale!(dtd, λ)
    axpy!(1.0, dtd, normalization)
    map!(x -> x > 0 ? 1 / sqrt(x) : 0, normalization, normalization)
    map!(sqrt, dtd, dtd)
    A = PreconditionedMatrix(DampenedMatrix(J, dtd), normalization, tmp)

    # solve
    x, ch = lsmr!(δx, A, b, v, h, hbar, btol = 0.5)
    return ch.mvps
end