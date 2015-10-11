
##############################################################################
## 
## solve J'J \ J'y by QR (used in Dogleg)
##
##############################################################################

type DenseQRSolver{Tqr, Tu <: AbstractVector} <: AbstractSolver
    qr::Tqr
    u::Tu
    function DenseQRSolver(qr, u)
        length(u) == size(qr, 1) || throw(DimensionMismatch("u must have length size(qr, 1)"))
        new(qr, u)
    end
end
DenseQRSolver{Tqr, Tu <: AbstractVector}(qr::Tqr, u::Tu) = DenseQRSolver{Tqr, Tu}(qr, u)

function allocate(nls::DenseLeastSquaresProblem,
    ::Type{Val{:dogleg}}, ::Type{Val{:factorization}})
    return DenseQRSolver(similar(nls.J), _zeros(nls.y))
end

function solve!(x, nls::DenseLeastSquaresProblem, solve::DenseQRSolver)
    y, J = nls.y, nls.J
    u, qr = solve.u, solve.qr
    
    copy!(qr, J)
    copy!(u, y)
    A_ldiv_B!(qrfact!(qr, Val{true}), u)

    @inbounds @simd for i in 1:length(x)
        x[i] = u[i]
    end
    return x, 1
end

##############################################################################
## 
## solve (J'J + λ dtd) \ J'y by QR (used in LevenbergMarquardt)
##
##############################################################################


type DenseQRDampenedSolver{Tqr, Tu <: AbstractVector} <: AbstractSolver
    qr::Tqr
    u::Tu
    function DenseQRDampenedSolver(qr, u)
        length(u) == size(qr, 1) || throw(DimensionMismatch("u must have length size(qr, 1)"))
        new(qr, u)
    end
end
DenseQRDampenedSolver{Tqr, Tu <: AbstractVector}(qr::Tqr, u::Tu) = DenseQRDampenedSolver{Tqr, Tu}(qr, u)

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
    clamp!(dtd, MIN_DIAGONAL, MAX_DIAGONAL)
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
    A_ldiv_B!(qrfact!(qr, Val{true}), u)

    @inbounds @simd for i in 1:length(x)
        x[i] = u[i]
    end
    return x, 1
end
