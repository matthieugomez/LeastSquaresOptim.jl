
##############################################################################
## 
## Dogleg : solve J'J \ J'y by QR
##
##############################################################################

type DenseQRDogleg{Tqr, Tu} <: AbstractSolver
    qr::Tqr
    u::Tu
end

function allocate(nls::DenseLeastSquaresProblem,
    ::Type{Val{:dogleg}}, ::Type{Val{:factorization}})
    return DenseQRDogleg(similar(nls.J), _zeros(nls.y))
end

function solve!{T <: DenseLeastSquaresProblem, Tmethod <: Dogleg, Tsolve <: DenseQRDogleg}(
    anls::LeastSquaresProblemAllocated{T, Tmethod, Tsolve})
    y, J = anls.nls.y, anls.nls.J
    u, qr = anls.solve.u, anls.solve.qr
    δgn = anls.method. δgn
    
    copy!(qr, J)
    copy!(u, y)
    A_ldiv_B!(qrfact!(qr), u)

    for i in 1:length(δgn)
        δgn[i] = u[i]
    end
    return 1
end

##############################################################################
## 
## LevenbergMarquardt: solve (J'J + λ dtd) \ J'y by QR
##
##############################################################################


type DenseQRLevenvergMarquardt{Tqr, Tu} <: AbstractSolver
    qr::Tqr
    u::Tu
end

function allocate(nls:: DenseLeastSquaresProblem,
    ::Type{Val{:levenberg_marquardt}}, ::Type{Val{:factorization}})
    qr = zeros(eltype(nls.J), length(nls.y) + length(nls.x), length(nls.x))
    u = zeros(length(nls.y) + length(nls.x))
    return DenseQRLevenvergMarquardt(qr, u)
end

function solve!{T <: DenseLeastSquaresProblem, Tmethod <: LevenbergMarquardt, Tsolve <: DenseQRLevenvergMarquardt}(
    anls::LeastSquaresProblemAllocated{T, Tmethod, Tsolve}, λ)
    y, J = anls.nls.y, anls.nls.J
    u, qr = anls.solve.u, anls.solve.qr
    dtd, δx = anls.method.dtd, anls.method.δx
    
    # transform dtd
    clamp!(dtd, MIN_DIAGONAL, Inf)
    scale!(dtd, λ)

    # update qr as |J; diagm(dtd)|
    fill!(qr, zero(eltype(qr)))
    @inbounds for j in 1:size(J, 2)
        @simd for i in 1:size(J, 1)
            qr[i, j] = J[i, j]
        end
    end
    leny = length(y)
    @inbounds for i in 1:length(dtd)
        qr[leny + i, i] = sqrt(dtd[i])
    end

    # update u as |J; 0|
    @inbounds @simd for i in 1:length(y)
        u[i] = y[i]
    end
    @inbounds @simd for i in (length(y)+1):length(u)
        u[i] = 0
    end
  
    # solve
    A_ldiv_B!(qrfact!(qr), u)

    for i in 1:length(δx)
        δx[i] = u[i]
    end
    return 1
end

##############################################################################
## 
## Dogleg : solve J'J \ J'y by Cholesky
##
##############################################################################

type CholeskyDenseDogleg{Tc} <: AbstractSolver
    chol::Tc
end

function allocate(nls::DenseLeastSquaresProblem,
    ::Type{Val{:dogleg}}, ::Type{Val{:factorization_cholesky}})
    return CholeskyDenseDogleg(Array(eltype(nls.J), length(nls.x), length(nls.x)))
end

function solve!{T <: DenseLeastSquaresProblem, Tmethod <: Dogleg, Tsolve <: CholeskyDenseDogleg}(
    anls::LeastSquaresProblemAllocated{T, Tmethod, Tsolve})
    y, J = anls.nls.y, anls.nls.J
    chol = anls.solve.chol
    δgn = anls.method. δgn
    
    At_mul_B!(chol, J,  J)
    At_mul_B!(δgn, J,  y)
    A_ldiv_B!(cholfact!(chol), δgn)
    return 1
end

##############################################################################
## 
## LevenbergMarquardt: solve (J'J + λ dtd) \ J'y by Cholesky
##
##############################################################################

type DenseLevenbergMarquardt{Tc} <: AbstractSolver
    chol::Tc
end

function allocate(nls:: DenseLeastSquaresProblem,
    ::Type{Val{:levenberg_marquardt}}, ::Type{Val{:factorization_cholesky}})
    return DenseLevenbergMarquardt(Array(eltype(nls.J), length(nls.x), length(nls.x)))
end

function solve!{T <: DenseLeastSquaresProblem, Tmethod <: LevenbergMarquardt, Tsolve <: DenseLevenbergMarquardt}(
    anls::LeastSquaresProblemAllocated{T, Tmethod, Tsolve}, λ)
    y, J = anls.nls.y, anls.nls.J
    chol = anls.solve.chol
    dtd, δx = anls.method.dtd, anls.method.δx
    
    # update chol as J'J + λdtd
    At_mul_B!(chol, J, J)
    clamp!(dtd, MIN_DIAGONAL, Inf)
    scale!(dtd, λ)
    @inbounds @simd for i in 1:size(chol, 1)
        chol[i, i] += dtd[i]
    end

    # solve
    At_mul_B!(δx, J, y)
    A_ldiv_B!(cholfact!(chol), δx)
    return 1
end

