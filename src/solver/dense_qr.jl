type DenseQRSolver{Tqr <: StridedMatrix, Tu <: AbstractVector} <: AbstractSolver
    qr::Tqr
    u::Tu
    function DenseQRSolver(qr, u)
        length(u) == size(qr, 1) || throw(DimensionMismatch("u must have length size(J, 1)"))
        new(qr, u)
    end
end

DenseQRSolver{Tqr, Tu <: AbstractVector}(qr::Tqr, u::Tu) = DenseQRSolver{Tqr, Tu}(qr, u)

##############################################################################
## 
## solve J'J \ J'y by QR (used in Dogleg)
##
##############################################################################

function AbstractSolver(nls::DenseLeastSquaresProblem,
    ::Type{Val{:dogleg}}, ::Type{Val{:qr}})
    return DenseQRSolver(similar(nls.J), _zeros(nls.y))
end

function solve!(x::AbstractVector, J::StridedMatrix,  y::AbstractVector, A::DenseQRSolver)
    u, qr = A.u, A.qr
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

function AbstractSolver(nls:: DenseLeastSquaresProblem,
    ::Type{Val{:levenberg_marquardt}}, ::Type{Val{:qr}})
    qr = zeros(eltype(nls.J), length(nls.y) + length(nls.x), length(nls.x))
    u = zeros(length(nls.y) + length(nls.x))
    return DenseQRSolver(qr, u)
end

function solve!(x::AbstractVector, J::StridedMatrix, y::AbstractVector, 
                dtd::AbstractVector, λ::Real, A::DenseQRSolver)
    u, qr = A.u, A.qr
    
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
