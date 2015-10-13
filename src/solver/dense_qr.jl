type DenseQRSolver{TJ, Tqr, Tu <: AbstractVector} <: AbstractOperator
    J::TJ
    qr::Tqr
    u::Tu
    function DenseQRSolver(J, qr, u)
        length(u) == size(qr, 1) || throw(DimensionMismatch("u must have length size(qr, 1)"))
        new(J, qr, u)
    end
end

DenseQRSolver{TJ, Tqr, Tu <: AbstractVector}(J::TJ, qr::Tqr, u::Tu) = DenseQRSolver{TJ, Tqr, Tu}(J, qr, u)

##############################################################################
## 
## solve J'J \ J'y by QR (used in Dogleg)
##
##############################################################################

function AbstractOperator(nls::DenseLeastSquaresProblem,
    ::Type{Val{:dogleg}}, ::Type{Val{:qr}})
    return DenseQRSolver(nls.J, similar(nls.J), _zeros(nls.y))
end

function solve!(x, A::DenseQRSolver, y)
    J, u, qr = A.J, A.u, A.qr
    
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

function AbstractOperator(nls:: DenseLeastSquaresProblem,
    ::Type{Val{:levenberg_marquardt}}, ::Type{Val{:qr}})
    qr = zeros(eltype(nls.J), length(nls.y) + length(nls.x), length(nls.x))
    u = zeros(length(nls.y) + length(nls.x))
    return DenseQRSolver(nls.J, qr, u)
end

function solve!(x, A::DenseQRSolver, y, dtd, λ)
    J, u, qr = A.J, A.u, A.qr
    
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
