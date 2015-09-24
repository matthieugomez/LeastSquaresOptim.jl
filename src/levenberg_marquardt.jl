##############################################################################
##
## levenberg_marquard method
##
## x a vector or any type that implements: norm, sumabs2, dot, similar, fill!, copy!, axpy!, map!
## fcur is a vector or any type that implements: sumabs2(fcur), scale!(fcur, α), similar(fcur), axpy!
## J is a matrix, a SparseMatrixSC, or anything that implements
## sumabs21(vec, J) : updates vec to sumabs(J, 1)
## A_mul_B!(α, A, b, β, c) updates c -> α Ab + βc
## Ac_mul_B!(α, A, b, β, c) updates c -> α A'b + βc
##
## f!(x, out) returns vector f_i(x) in in out
## g!(x, out) returns jacobian in out
##
## x is the initial solution and is transformed in place to the solution
##
##############################################################################

const MAX_λ = 1e16 # minimum trust region radius
const MIN_λ = 1e-16 # maximum trust region radius
const MIN_STEP_QUALITY = 1e-3
const GOOD_STEP_QUALITY = 0.75
const MIN_DIAGONAL = 1e-6

function levenberg_marquardt!(x, fcur, f!, J, g!; 
                              tol::Number = 1e-8, maxiter::Integer = 1000, λ::Number = 10.0)

    # temporary array
    δx = similar(x)
    dtd = similar(x)
    ftrial = similar(fcur)
    f_predict = similar(fcur)

    # for ls
    alloc = lm_lssolver_alloc(x, fcur, J)

    # initialize
    f!(x, fcur)
    mfcur = scale!(fcur, -1.0)
    residual = sumabs2(mfcur)
    need_jacobian = true

    iter = 0
    while iter < maxiter 
        iter += 1
        if need_jacobian
            g!(x, J)
            need_jacobian = false
        end
        sumabs21!(dtd, J)
        # solve (J'J + λ * diagm(dtd)) = -J'fcur
        currentiter = lm_lssolver!(δx, mfcur, J, dtd, λ, alloc)
        iter += currentiter
        # trial residual
        axpy!(1.0, δx, x)
        f!(x, ftrial)
        trial_residual = sumabs2(ftrial)
        
        if abs(residual - trial_residual) <= max(tol^2 * residual, eps()^2)
            return iter, true
        end

        # predicted residual
        A_mul_B!(f_predict, J, δx)
        axpy!(-1.0, mfcur, f_predict)
        predicted_residual = sumabs2(f_predict)
        ρ = (residual - trial_residual) / (residual - predicted_residual)
        if ρ > MIN_STEP_QUALITY
            scale!(mfcur, ftrial, -1.0)
            residual = trial_residual
            # increase trust region radius
            if ρ > GOOD_STEP_QUALITY
                λ = max(0.1 * λ, MIN_λ)
            end
            need_jacobian = true
        else
            # revert update
            axpy!(-1.0, δx, x)
            λ = min(10 * λ, MAX_λ)
        end
    end
    return maxiter, false
end

##############################################################################
## 
## Case of Dense Matrix
##
##############################################################################

function sumabs21!{T}(dtd::Vector{T}, J::Matrix{T})
    for i in 1:length(dtd)
        dtd[i] = sumabs2(slice(J, :, i))
    end
end

function lm_lssolver_alloc{T}(x::Vector{T}, mfcur::Vector{T}, J::Matrix{T})
    M = Array(T, length(x), length(x))
    rhs = Array(T, length(x))
    return M, rhs
end

function lm_lssolver!{T}(δx::Vector{T}, mfcur::Vector{T}, J::Matrix{T}, dtd::Vector{T}, λ, alloc)
 
    M, rhs = alloc

    # update M as J'J + λ^2dtd
    At_mul_B!(M, J, J)
    clamp!(dtd, MIN_DIAGONAL, Inf)
    scale!(dtd, λ^2)
    for i in 1:size(M, 1)
        M[i, i] += dtd[i]
    end

    # update rhs as J' fcur
    At_mul_B!(rhs, J, mfcur)

    # solve
    δx[:] = M \ rhs
    return 1
end

##############################################################################
## 
## Case where J'J is costly to store: Sparse Matrix, or anything
## that defines two functions
## A_mul_B(α, A, a, β b) that updates b as α A a + β b 
## Ac_mul_B(α, A, a, β b) that updates b as α A' a + β b 
## sumabs21(x, A) that updates x with sumabs2(A)
##
## we use LSMR for the problem ||Ax-b|| with
## matrix A = |J         |
##            |diag(dtd) |
## and 1/sqrt(diag(A'A)) as preconditioner
##
## We only solve with btol = 0.5
## See "An Inexact Levenberg-Marquardt Method for Large Sparse Nonlinear Least Squares"
## Weight Holt (1985)
##############################################################################

type MatrixWrapper{TA, Tx}
    A::TA # J
    d::Tx # λ * sqrt(diag(J'J))
    normalization::Tx # 1/sqrt((1 + λ^2) diag(J'J)))
    tmp::Tx
end

type VectorWrapper{Ty, Tx}
    y::Ty # dimension of f(x)
    x::Tx # dimension of x
end

# These functions are used in lsmr (ducktyping)
function copy!{Ty, Tx}(a::VectorWrapper{Ty, Tx}, b::VectorWrapper{Ty, Tx})
    copy!(a.y, b.y)
    copy!(a.x, b.x)
    return a
end

function fill!(a::VectorWrapper, α::Number)
    fill!(a.y, α)
    fill!(a.x, α)
    return a
end

function scale!(a::VectorWrapper, α::Number)
    scale!(a.y, α)
    scale!(a.x, α)
    return a
end

function axpy!{Ty, Tx}(α::Number, a::VectorWrapper{Ty, Tx}, b::VectorWrapper{Ty, Tx})
    axpy!(α, a.y, b.y)
    axpy!(α, a.x, b.x)
    return b
end

function norm(a::VectorWrapper)
    return sqrt(norm(a.y)^2 + norm(a.x)^2)
end

function A_mul_B!{TA, Tx, Ty}(α::Number, mw::MatrixWrapper{TA, Tx}, a::Tx, 
                β::Number, b::VectorWrapper{Ty, Tx})
    if β != 1.
        if β == 0.
            fill!(b, 0.)
        else
            scale!(b, β)
        end
    end
    map!((x, z) -> x * z, mw.tmp, a, mw.normalization)
    A_mul_B!(α, mw.A, mw.tmp, 1.0, b.y)
    map!((z, x, y)-> z + α * x * y, b.x, b.x, mw.tmp, mw.d)
    return b
end

function Ac_mul_B!{TA, Tx, Ty}(α::Number, mw::MatrixWrapper{TA, Tx}, a::VectorWrapper{Ty, Tx}, 
                β::Number, b::Tx)
    Ac_mul_B!(α, mw.A, a.y, 0.0, mw.tmp)
    map!((z, x, y)-> z + α * x * y, mw.tmp, mw.tmp, a.x, mw.d)
    map!((x, z) -> x * z, mw.tmp, mw.tmp, mw.normalization)
    if β != 1.
        if β == 0.
            fill!(b, 0.)
        else
            scale!(b, β)
        end
    end
    axpy!(1.0, mw.tmp, b)
    return b
end

function lm_lssolver_alloc(x, fcur, J)
    normalization = similar(x)
    tmp = similar(x)
    fill!(tmp, zero(Float64))
    zerosvector = similar(x)
    fill!(zerosvector, zero(Float64))
    u = VectorWrapper(similar(fcur), similar(x))
    v = similar(x)
    h = similar(x)
    hbar = similar(x)
    return normalization, tmp, zerosvector, u, v, h, hbar
end

function lm_lssolver!(δx, mfcur, J, dtd, λ, alloc)
    # we use LSMR with the matrix A = |J         |
    #                                 |diag(dtd) |
    # and 1/sqrt(diag(A'A)) as preconditioner
    normalization, tmp, zerosvector, u, v, h, hbar = alloc
    copy!(normalization, dtd)
    map!(x -> max(x, MIN_DIAGONAL), dtd, dtd)
    scale!(dtd, λ^2)
    axpy!(1.0, dtd, normalization)
    map!(x -> x == 0. ? 0. : 1 / sqrt(x), normalization, normalization)
    map!(sqrt, dtd, dtd)
    y = VectorWrapper(mfcur, zerosvector)
    A = MatrixWrapper(J, dtd, normalization, tmp)
    fill!(δx, zero(Float64))
    iter = lsmr!(δx, y, A, u, v, h, hbar, btol = 0.5)
    map!((x, z) -> x * z, δx, δx, normalization)
    return iter
end


##############################################################################
##
## Particular case of Sparse matrix
##
##############################################################################


function sumabs21!(v::Vector, A::Base.SparseMatrix.SparseMatrixCSC)
    for i in 1:length(v)
        v[i] = sumabs2(sub(nonzeros(A), nzrange(A, i)))
    end
end