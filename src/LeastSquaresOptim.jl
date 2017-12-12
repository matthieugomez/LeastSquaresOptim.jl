__precompile__(true)

module LeastSquaresOptim

##############################################################################
##
## Dependencies
##
##############################################################################

import Base: A_mul_B!, Ac_mul_B!, A_ldiv_B!, copy!, fill!, scale!, norm, axpy!, eltype, length, size
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
