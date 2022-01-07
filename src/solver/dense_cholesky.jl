##############################################################################
## 
## Type with stored cholesky
##
##############################################################################

struct DenseCholeskyAllocatedSolver{Tc <: StridedMatrix} <: AbstractAllocatedSolver
    cholm::Tc
    function DenseCholeskyAllocatedSolver{Tc}(cholm) where {Tc <: StridedMatrix}
        size(cholm, 1) == size(cholm, 2) || throw(DimensionMismatch("chol must be square"))
        new(cholm)
    end
end
function DenseCholeskyAllocatedSolver(cholm::Tc) where {Tc <: StridedMatrix}
    DenseCholeskyAllocatedSolver{Tc}(cholm)
end


function AbstractAllocatedSolver(nls::LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}, optimizer::AbstractOptimizer{Cholesky}) where {Tx, Ty, Tf, TJ <: StridedVecOrMat, Tg}
    return DenseCholeskyAllocatedSolver(Array{eltype(nls.J)}(undef, length(nls.x), length(nls.x)))
end

##############################################################################
## 
## solve J'J \ J'y by Cholesky
##
##############################################################################

function LinearAlgebra.ldiv!(x::AbstractVector, J::StridedMatrix, y::AbstractVector, A::DenseCholeskyAllocatedSolver)
    cholm = A.cholm
    mul!(cholm, J',  J)
    mul!(x, J',  y)
    ldiv!(cholesky!(Symmetric(cholm), Val(true)), x)
    return x, 1
end

##############################################################################
## 
## solve (J'J + diagm(damp)) \ J'y by Cholesky
##
##############################################################################

function LinearAlgebra.ldiv!(x::AbstractVector, J::StridedMatrix, y::AbstractVector, 
            damp::AbstractVector, A::DenseCholeskyAllocatedSolver)
    cholm = A.cholm
    
    # update cholm as J'J + λdtd
    mul!(cholm, J', J)
    # transform dammp
    size(cholm, 1) == length(damp) || throw(DimensionMismatch("size(chol, 1) should equal length(damp)"))
   for i in 1:size(cholm, 1)
        cholm[i, i] += damp[i]
    end

    # solve
    mul!(x, J', y)
    ldiv!(cholesky!(Symmetric(cholm)), x)
    return x, 1
end

