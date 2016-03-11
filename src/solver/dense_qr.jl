##############################################################################
## 
## Type with stored qr
##
##############################################################################

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
## solve J'J \ J'y by QR
##
##############################################################################

function AbstractSolver(nls::DenseLeastSquaresProblem,
    ::Type{Val{:dogleg}}, ::Type{Val{:qr}})
    return DenseQRSolver(similar(nls.J), _zeros(nls.y))
end

function A_ldiv_B!(x::AbstractVector, J::StridedMatrix,  y::AbstractVector, A::DenseQRSolver)
    u, qr = A.u, A.qr
    copy!(qr, J)
    copy!(u, y)
    A_ldiv_B!(qrfact!(qr, Val{true}), u)

    for i in 1:length(x)
        x[i] = u[i]
    end
    return x, 1
end

##############################################################################
## 
## solve (J'J + diagm(damp)) \ J'y by QR
##
##############################################################################

function AbstractSolver(nls:: DenseLeastSquaresProblem,
    ::Type{Val{:levenberg_marquardt}}, ::Type{Val{:qr}})
    qr = zeros(eltype(nls.J), length(nls.y) + length(nls.x), length(nls.x))
    u = zeros(length(nls.y) + length(nls.x))
    return DenseQRSolver(qr, u)
end

function A_ldiv_B!(x::AbstractVector, J::StridedMatrix, y::AbstractVector, 
                damp::AbstractVector, A::DenseQRSolver, verbose::Bool = false)
    u, qr = A.u, A.qr
    
    # transform dammp
    length(u) ==  length(y) + length(x) || throw(DimensionMismatch("length(u) should equal length(x) + length(y)"))

    # update qr as |J; diagm(damp)|
    fill!(qr, zero(eltype(qr)))
    for j in 1:size(J, 2)
        for i in 1:size(J, 1)
            qr[i, j] = J[i, j]
        end
    end

    leny = length(y)
    for i in 1:length(damp)
        qr[leny + i, i] = sqrt(damp[i])
    end

    # update u as |J; 0|
    fill!(u, zero(eltype(u)))
    for i in 1:length(y)
        u[i] = y[i]
    end
    if verbose
        @show mean(u)
        sleep(1)
    end

    # solve
    A_ldiv_B!(qrfact!(qr, Val{true}), u)
    if verbose
        @show mean(u)
        sleep(1)
    end
    for i in 1:length(x)
        x[i] = u[i]
    end
    return x, 1
end
