module LeastSquaresOptim

##############################################################################
##
## Dependencies
##
##############################################################################

import Base: copyto!, fill!, eltype, length, size
import LinearAlgebra: mul!, rmul!, norm, cholesky!, qr!, Symmetric, dot, eigen, axpy!, svd, ldiv!, Transpose, adjoint, Adjoint
import LinearAlgebra.BLAS: gemm!
import Printf: @printf, @sprintf
if Base.USE_GPL_LIBS
    import SuiteSparse.SPQR: QRSparse
    import SparseArrays: SparseMatrixCSC, sparse, nzrange, nonzeros
end
import Statistics: mean
using ForwardDiff
import Optim: optimize

##############################################################################
##
## Exported methods and types 
##
##############################################################################

export optimize!, 
LeastSquaresProblem,
LeastSquaresProblemAllocated,
LeastSquaresResult,
Dogleg,
LevenbergMarquardt,
optimize

##############################################################################
##
## Load files
##
##############################################################################

include("utils/lsmr.jl")
include("utils/utils.jl")

include("types.jl")
include("optimizer/levenberg_marquardt.jl")
include("optimizer/dogleg.jl")

include("solver/dense_qr.jl")
include("solver/dense_cholesky.jl")
include("solver/iterative_lsmr.jl")

end
