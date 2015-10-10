
##############################################################################
## 
## solve J'J \ J'y by QR (used in Dogleg)
##
##############################################################################

type DenseQRSolver{Tqr, Tu} <: AbstractSolver
    qr::Tqr
    u::Tu
end

function allocate(nls::DenseLeastSquaresProblem,
    ::Type{Val{:dogleg}}, ::Type{Val{:factorization}})
    return DenseQRSolver(similar(nls.J), _zeros(nls.y))
end

function solve!(x, nls::DenseLeastSquaresProblem, solve::DenseQRSolver)
    y, J = nls.y, nls.J
    u, qr = solve.u, solve.qr
    
    copy!(qr, J)
    copy!(u, y)
    A_ldiv_B!(qrfact!(qr), u)

    @inbounds @simd for i in 1:length(x)
        x[i] = u[i]
    end
    return 1
end

##############################################################################
## 
## solve (J'J + λ dtd) \ J'y by QR (used in LevenbergMarquardt)
##
##############################################################################


type DenseQRDampenedSolver{Tqr, Tu} <: AbstractSolver
    qr::Tqr
    u::Tu
end

function allocate(nls:: DenseLeastSquaresProblem,
    ::Type{Val{:levenberg_marquardt}}, ::Type{Val{:factorization}})
    qr = zeros(eltype(nls.J), length(nls.y) + length(nls.x), length(nls.x))
    u = zeros(length(nls.y) + length(nls.x))
    return DenseQRDampenedSolver(qr, u)
end

function solve!(x, dtd, λ, nls::DenseLeastSquaresProblem, solve::DenseQRDampenedSolver)
    y, J = nls.y, nls.J
    u, qr = solve.u, solve.qr
    
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
    fill!(u, zero(eltype(u)))
    @inbounds @simd for i in 1:length(y)
        u[i] = y[i]
    end

    # solve
    A_ldiv_B!(qrfact!(qr), u)

    @inbounds @simd for i in 1:length(x)
        x[i] = u[i]
    end
    return 1
end

##############################################################################
## 
## solve J'J \ J'y by Cholesky (used in Dogleg)
##
##############################################################################

type DenseCholeskySolver{Tc} <: AbstractSolver
    chol::Tc
end

function allocate(nls::DenseLeastSquaresProblem,
    ::Type{Val{:dogleg}}, ::Type{Val{:factorization_cholesky}})
    return DenseCholeskySolver(Array(eltype(nls.J), length(nls.x), length(nls.x)))
end

function solve!(x, nls::DenseLeastSquaresProblem, solve::DenseCholeskySolver)
    y, J = nls.y, nls.J
    chol = solve.chol
    
    Ac_mul_B!(chol, J,  J)
    Ac_mul_B!(x, J,  y)
    A_ldiv_B!(cholfact!(chol, :U, Val{true}), x)
    return 1
end

##############################################################################
## 
## solve (J'J + λ dtd) \ J'y by Cholesky (used in LevenbergMarquardt)
##
##############################################################################

type DenseCholeskyDampenedSolver{Tc} <: AbstractSolver
    chol::Tc
end

function allocate(nls:: DenseLeastSquaresProblem,
    ::Type{Val{:levenberg_marquardt}}, ::Type{Val{:factorization_cholesky}})
    return DenseCholeskyDampenedSolver(Array(eltype(nls.J), length(nls.x), length(nls.x)))
end

function solve!(x, dtd, λ, nls::DenseLeastSquaresProblem, solve::DenseCholeskyDampenedSolver)
    y, J = nls.y, nls.J
    chol = solve.chol
    
    # update chol as J'J + λdtd
    Ac_mul_B!(chol, J, J)
    clamp!(dtd, MIN_DIAGONAL, Inf)
    scale!(dtd, λ)
    @inbounds @simd for i in 1:size(chol, 1)
        chol[i, i] += dtd[i]
    end

    # solve
    Ac_mul_B!(x, J, y)
    A_ldiv_B!(cholfact!(chol), x)
    return 1
end

