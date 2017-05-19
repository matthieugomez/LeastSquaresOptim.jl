##############################################################################
## 
## Type with stored qr
##
##############################################################################
struct DenseQRAllocatedSolver{Tqr <: StridedMatrix, Tu <: AbstractVector} <: AbstractAllocatedSolver
    qr::Tqr
    u::Tu
    function DenseQRAllocatedSolver{Tqr, Tu}(qr, u) where {Tqr <: StridedMatrix, Tu <: AbstractVector}
        length(u) == size(qr, 1) || throw(DimensionMismatch("u must have length size(J, 1)"))
        new(qr, u)
    end
end

function DenseQRAllocatedSolver(qr::Tqr, u::Tu) where {Tqr <: StridedMatrix, Tu <: AbstractVector}
    DenseQRAllocatedSolver{Tqr, Tu}(qr, u)
end

##############################################################################
## 
## solve J'J \ J'y by QR
##
##############################################################################

function AbstractAllocatedSolver(nls::LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}, optimizer::Dogleg, solver::QR) where {Tx, Ty, Tf, TJ <: StridedVecOrMat, Tg}
    return DenseQRAllocatedSolver(similar(nls.J), _zeros(nls.y))
end

function A_ldiv_B!(x::AbstractVector, J::StridedMatrix,  y::AbstractVector, A::DenseQRAllocatedSolver)
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

function AbstractAllocatedSolver(nls::LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}, optimizer::LevenbergMarquardt, solver::QR) where {Tx, Ty, Tf, TJ <: StridedVecOrMat, Tg}
    qr = zeros(eltype(nls.J), length(nls.y) + length(nls.x), length(nls.x))
    u = zeros(length(nls.y) + length(nls.x))
    return DenseQRAllocatedSolver(qr, u)
end

function A_ldiv_B!(x::AbstractVector, J::StridedMatrix, y::AbstractVector, 
                damp::AbstractVector, A::DenseQRAllocatedSolver, verbose::Bool = false)
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
