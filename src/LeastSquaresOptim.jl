module LeastSquaresOptim

##############################################################################
##
## Dependencies
##
##############################################################################

using LinearAlgebra: LinearAlgebra, mul!, rmul!, norm, cholesky!, qr!, Symmetric, dot, eigen, axpy!, svd, ldiv!, Transpose, adjoint, Adjoint, ColumnNorm
using Printf: @printf, @sprintf
using SparseArrays: SparseMatrixCSC, sparse, nzrange, nonzeros
using ForwardDiff: JacobianConfig, Chunk, checktag, jacobian!
using FiniteDiff: JacobianCache, finite_difference_jacobian!
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
