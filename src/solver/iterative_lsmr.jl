#############################################################################
## 
## Type with stored vectors for lsmr
##
##############################################################################

type LSMRSolver{Tx1, Tx2, Tx3, Tx4, Tx5, Tx6, Ty} <: AbstractSolver
    normalization::Tx1
    tmp::Tx2
    v::Tx3
    h::Tx4
    hbar::Tx5
    zerosvector::Tx6
    u::Ty
    function LSMRSolver(normalization, tmp, v, h, hbar, zerosvector, u)
        length(normalization) == length(tmp) || throw(DimensionMismatch("normalization and tmp must have the same length"))
        length(normalization) == length(v) || throw(DimensionMismatch("normalization and v must have the same length"))
        length(normalization) == length(h) || throw(DimensionMismatch("normalization and h must have the same length"))
        length(normalization) == length(hbar) || throw(DimensionMismatch("normalization and hbar must have the same length"))
        length(normalization) == length(zerosvector) || throw(DimensionMismatch("normalization and zerosvector must have the same length"))
        new(normalization, tmp, v, h, hbar, zerosvector, u)
    end
end

function LSMRSolver{Tx1, Tx2, Tx3, Tx4, Tx5, Tx6, Ty}(normalization::Tx1, tmp::Tx2, v::Tx3, h::Tx4, hbar::Tx5, zerosvector::Tx6, u::Ty)
    LSMRSolver{Tx1, Tx2, Tx3, Tx4, Tx5, Tx6, Ty}(normalization, tmp, v, h, hbar, zerosvector, u)
end

function AbstractSolver(nls::LeastSquaresProblem, ::Type,
     ::Type{Val{:iterative}})
    LSMRSolver(_zeros(nls.x), _zeros(nls.x), _zeros(nls.x), 
        _zeros(nls.x), _zeros(nls.x),  _zeros(nls.x), _zeros(nls.y))
end

#############################################################################
## 
## solve J'J \ J'y
##
## we use LSMR on A / sqrt(diag(J'J)) (diagonal preconditioner)
## LSMR works on any matrix with the following methods:
## eltype, size, A_mul_B!, Ac_mul_B!
##
##############################################################################

type PreconditionedMatrix{TA, Tx}
    A::TA
    normalization::Tx 
    tmp::Tx
end
eltype(A::PreconditionedMatrix) = eltype(A.A)
size(A::PreconditionedMatrix, i::Integer) = size(A.A, i)
function A_mul_B!{TA, Tx}(α::Number, pm::PreconditionedMatrix{TA, Tx}, a::Tx, 
                β::Number, b)
    map!(*, pm.tmp, a, pm.normalization)
    A_mul_B!(α, pm.A, pm.tmp, β, b)
    return b
end
function Ac_mul_B!{TA, Tx}(α::Number, pm::PreconditionedMatrix{TA, Tx}, a, 
                β::Number, b::Tx)
    T = eltype(b)
    β = convert(T, β)
    Ac_mul_B!(one(T), pm.A, a, zero(T), pm.tmp)
    map!(*, pm.tmp, pm.tmp, pm.normalization)
    if β != one(T)
        if β == zero(T)
            fill!(b, β)
        else
            scale!(b, β)
        end
    end
    axpy!(α, pm.tmp, b)
    return b
end

function A_ldiv_B!(x, J, y, A::LSMRSolver)
    normalization, tmp, v, h, hbar, u = A.normalization, A.tmp, A.v, A.h, A.hbar, A.u

    # prepare x
    fill!(x, 0)

    # prepare b
    copy!(u, y)

    # prepare A
    colsumabs2!(normalization, J)
    Tx = eltype(normalization)
    map!(x -> x > zero(Tx) ? 1 / sqrt(x) : zero(Tx), normalization, normalization)
    A = PreconditionedMatrix(J, normalization, tmp)

    # solve
    x, ch = lsmr!(x, A, u, v, h, hbar)
    map!(*, x, x, A.normalization)
    return x, ch.mvps
end

##############################################################################
## 
## solve (J'J + damp I) \ J'y (used in LevenbergMarquardt)
## No need to solve exactly :
## "An Inexact Levenberg-Marquardt Method for Large Sparse Nonlinear Least Squares"
## Weight Holt (1985)
##
## We use
## LSMR with matrix A = |J         |
##                      |diag(dtd) |
## + diagonal preconditioner
## LSMR works on any matrix with the following methods:
## eltype, size, A_mul_B!, Ac_mul_B!
## LSMR works on any vector with the following methods:
## eltype, length, scale!, norm
##
##############################################################################

# First define a new type for this augmented space
# and methods use within lsmr! on this new space

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
        scale!(b, β)
    end
    A_mul_B!(α, mw.A, a, 1, b.y)
    map!((z, x, y)-> z + α * x * y, b.x, b.x, a, mw.diagonal)
    return b
end
function Ac_mul_B!{TA, Tx, Ty}(α::Number, mw::DampenedMatrix{TA, Tx}, a::DampenedVector{Ty, Tx}, 
                β::Number, b::Tx)
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

function A_ldiv_B!(x, J, y, damp, A::LSMRSolver)
    normalization, tmp, v, h, hbar, zerosvector, u = 
            A.normalization, A.tmp, A.v, A.h, A.hbar, A.zerosvector, A.u
    
    # prepare x
    fill!(x, 0)

    # prepare b
    copy!(u, y)
    fill!(zerosvector, 0)
    b = DampenedVector(u, zerosvector)

    # prepare A
    fill!(tmp, 0)
    colsumabs2!(normalization, J)
    Tx = eltype(normalization)
    axpy!(one(Tx), damp, normalization)
    map!(x -> x > zero(Tx) ? 1 / sqrt(x) : zero(Tx), normalization, normalization)
    map!(sqrt, damp, damp)
    A = PreconditionedMatrix(DampenedMatrix(J, damp), normalization, tmp)

    # solve
    x, ch = lsmr!(x, A, b, v, h, hbar, btol = 0.5)
    map!(*, x, x, A.normalization)
    return x, ch.mvps
end