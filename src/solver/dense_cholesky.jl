##############################################################################
## 
## Type with stored cholesky
##
##############################################################################

type DenseCholeskyAllocatedSolver{Tc <: StridedMatrix} <: AbstractAllocatedSolver
    chol::Tc
    function DenseCholeskyAllocatedSolver(chol)
        size(chol, 1) == size(chol, 2) || throw(DimensionMismatch("chol must be square"))
        new(chol)
    end
end

DenseCholeskyAllocatedSolver{Tc}(chol::Tc) = DenseCholeskyAllocatedSolver{Tc}(chol)
function AbstractAllocatedSolver{Tx, Ty, Tf, TJ <: StridedVecOrMat, Tg}(nls::LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}, optimizer, ::Cholesky)
    return DenseCholeskyAllocatedSolver(Array(eltype(nls.J), length(nls.x), length(nls.x)))
end

##############################################################################
## 
## solve J'J \ J'y by Cholesky
##
##############################################################################

function A_ldiv_B!(x::AbstractVector, J::StridedMatrix, y::AbstractVector, A::DenseCholeskyAllocatedSolver)
    chol = A.chol
    Ac_mul_B!(chol, J,  J)
    Ac_mul_B!(x, J,  y)
    A_ldiv_B!(cholfact!(chol, :U, Val{true}), x)
    return x, 1
end

##############################################################################
## 
## solve (J'J + diagm(damp)) \ J'y by Cholesky
##
##############################################################################

function A_ldiv_B!(x::AbstractVector, J::StridedMatrix, y::AbstractVector, 
            damp::AbstractVector, A::DenseCholeskyAllocatedSolver)
    chol = A.chol
    
    # update chol as J'J + λdtd
    Ac_mul_B!(chol, J, J)
    # transform dammp
    size(chol, 1) == length(damp) || throw(DimensionMismatch("size(chol, 1) should equal length(damp)"))
   for i in 1:size(chol, 1)
        chol[i, i] += damp[i]
    end

    # solve
    Ac_mul_B!(x, J, y)
    A_ldiv_B!(cholfact!(chol), x)
    return x, 1
end

