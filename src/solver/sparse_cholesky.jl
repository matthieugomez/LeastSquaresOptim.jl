##############################################################################
## 
## Utils
##
##############################################################################
if VERSION < v"0.6.0-"

    # Update B as Sparse(A)
    function sparse!{Tv<:VTypes,Ti<:ITypes}(A::SparseMatrixCSC{Tv,Ti}, B::Sparse{Tv})
        s = unsafe_load(B.p)
        unsafe_copy!(s.x, pointer(A.nzval), length(A.nzval))
        check_sparse(B) == Int32(1) || throw(CHOLMODException(""))
        return B
    end

    # Update B as A'
    function transpose!{Tv<: VTypes}(A::Sparse{Tv}, B::Sparse{Tv})
        out = ccall((@cholmod_name("transpose_unsym", SuiteSparse_long),:libcholmod),
            Cint,
                (Ptr{C_Sparse{Tv}}, Cint, Ptr{SuiteSparse_long}, Ptr{SuiteSparse_long}, Csize_t, Ptr{C_Sparse{Tv}}, Ptr{UInt8}),   
                A.p, 2, C_NULL, C_NULL, 0, B.p, common())
        out == Int32(1) || throw(CHOLMODException(""))
        return B
    end
end


##############################################################################
## 
## Constructor
##
##############################################################################

type SparseCholeskyAllocatedSolver{Tv, Ti <: Integer, Tx <: AbstractVector} <: AbstractAllocatedSolver
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    v::Tx
    sparseJ::Sparse{Tv}
    sparseJt::Sparse{Tv}
    F::Factor{Tv}
    cm::Array{UInt8, 1}
    function SparseCholeskyAllocatedSolver(colptr, rowval, v, sparseJ, sparseJt, F, cm)
        size(sparseJ) == (size(sparseJt, 2),  size(sparseJt, 1)) || throw(DimensionMismatch("J and Jt should have transposed dimension"))
        new(colptr, rowval, v, sparseJ, sparseJt, F, cm)
    end
end

function SparseCholeskyAllocatedSolver{Tv, Ti <: Integer, Tx <: AbstractVector}(colptr::Vector{Ti}, rowval::Vector{Ti}, v::Tx, sparseJ::Sparse{Tv}, sparseJt::Sparse{Tv}, F::Factor{Tv}, cm::Array{UInt8, 1})
    SparseCholeskyAllocatedSolver{Tv, Ti, Tx}(colptr, rowval, v, sparseJ, sparseJt, F, cm)
end

function AbstractAllocatedSolver{Tx, Ty, Tf, TJ <: SparseMatrixCSC , Tg}(nls::LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}, optimizer::Dogleg, solver::Cholesky)
    colptr = deepcopy(nls.J.colptr)
    rowval = deepcopy(nls.J.rowval)
    sparseJ = Sparse(nls.J)
    sparseJt = transpose_(sparseJ, 2)
    cm = defaults(common()) 
    set_print_level(cm, 0)
    unsafe_store!(common_final_ll, 1)
    F = analyze(sparseJt, cm)
    return SparseCholeskyAllocatedSolver(colptr, rowval, _zeros(nls.x), sparseJ, sparseJt, F, cm)
end

##############################################################################
## 
## solve J'J \ J'y by Cholesky
##
##############################################################################

function A_ldiv_B!(x::AbstractVector, J::SparseMatrixCSC, y::AbstractVector, A::SparseCholeskyAllocatedSolver)
    colptr, rowval, v, sparseJ, sparseJt, F, cm = 
    A.colptr, A.rowval, A.v, A.sparseJ, A.sparseJt, A.F, A.cm

    # check symbolic structure is the same
    if colptr != J.colptr || rowval != J.rowval
        error("The symbolic structure of the Jacobian has been changed. Either (i) rewrite g! so that it does not modify the structure of J (see Julia issue #9906) (ii) use solver = :iterative rather than solver = :cholesky")
    end

    sparse!(J, sparseJ)
    transpose!(sparseJ, sparseJt)
    factorize_p!(sparseJt, 0, F, cm)
    Ac_mul_B!(v, J, y)
    # !! there is a memory allocation here
    copy!(x, F \ v)
    return x, 1
end