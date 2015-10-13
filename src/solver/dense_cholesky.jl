type DenseCholeskyOperator{TJ <: StridedMatrix, Tc <: StridedMatrix} <: AbstractOperator
    J::TJ
    chol::Tc
    function DenseCholeskyOperator(J, chol)
        size(J, 2) == size(chol, 1) || throw(DimensionMismatch("J and chol must have the same size"))
        size(chol, 1) == size(chol, 2) || throw(DimensionMismatch("chol must be square"))
        new(J, chol)
    end
end

DenseCholeskyOperator{TJ, Tc}(J::TJ, chol::Tc) = DenseCholeskyOperator{TJ, Tc}(J, chol)
function AbstractOperator(nls:: DenseLeastSquaresProblem, ::Type, ::Type{Val{:cholesky}})
    return DenseCholeskyOperator(nls.J, Array(eltype(nls.J), length(nls.x), length(nls.x)))
end

##############################################################################
## 
## solve J'J \ J'y by Cholesky (used in Dogleg)
##
##############################################################################

function solve!(x::AbstractVector, A::DenseCholeskyOperator, y::AbstractVector)
    J, chol = A.J, A.chol
    Ac_mul_B!(chol, J,  J)
    Ac_mul_B!(x, J,  y)
    A_ldiv_B!(cholfact!(chol, :U, Val{true}), x)
    return x, 1
end

##############################################################################
## 
## solve (J'J + λ dtd) \ J'y by Cholesky (used in LevenbergMarquardt)
##
##############################################################################

function solve!(x::AbstractVector, A::DenseCholeskyOperator, y::AbstractVector, dtd::AbstractVector, λ)
    J, chol = A.J, A.chol
    
    # update chol as J'J + λdtd
    Ac_mul_B!(chol, J, J)
    clamp!(dtd, MIN_DIAGONAL, MAX_DIAGONAL)
    scale!(dtd, λ)
    @inbounds @simd for i in 1:size(chol, 1)
        chol[i, i] += dtd[i]
    end

    # solve
    Ac_mul_B!(x, J, y)
    A_ldiv_B!(cholfact!(chol), x)
    return x, 1
end

