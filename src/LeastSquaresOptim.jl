__precompile__(true)

module LeastSquaresOptim

##############################################################################
##
## Dependencies
##
##############################################################################

import Base: A_mul_B!, Ac_mul_B!, copy!, fill!, scale!, norm, axpy!, eltype, length, size
import Base.SparseMatrix.CHOLMOD: VTypes, ITypes, Sparse, Factor, C_Sparse, SuiteSparse_long, transpose_, @cholmod_name, common, defaults, set_print_level, common_final_ll, analyze, factorize_p!, check_sparse
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
include("method/levenberg_marquardt.jl")
include("method/dogleg.jl")
include("method/utils.jl")

include("solver/factorization_dense_qr.jl")
include("solver/factorization_dense_cholesky.jl")
include("solver/factorization_sparse.jl")
include("solver/iterative_lsmr.jl")

end
