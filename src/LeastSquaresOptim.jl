__precompile__(true)

module LeastSquaresOptim

##############################################################################
##
## Dependencies
##
##############################################################################

import Base: A_mul_B!, Ac_mul_B!, A_ldiv_B!, copy!, fill!, scale!, norm, axpy!, eltype, length, size, call
import Base.SparseArrays.CHOLMOD: VTypes, ITypes, Sparse, Factor, C_Sparse, SuiteSparse_long, transpose_, @cholmod_name, common, defaults, set_print_level, common_final_ll, analyze, factorize_p!, check_sparse
using ForwardDiff

##############################################################################
##
## Exported methods and types 
##
##############################################################################

export optimize!, 
LeastSquaresProblem,
LeastSquaresProblemAllocated,
LeastSquaresResult

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
include("solver/sparse_cholesky.jl")
include("solver/iterative_lsmr.jl")

end
