
##############################################################################
## 
## Utils
##
##############################################################################

macro isok(A)
    :($A ==  Int32(1) || throw(CHOLMODException("")))
end

# Update B as A'
function transpose_unsym_!{Tv<: VTypes}(A::Sparse{Tv}, values::Integer, B::Sparse{Tv})
    @isok ccall((@cholmod_name("transpose_unsym", SuiteSparse_long),:libcholmod),
        Cint,
            (Ptr{C_Sparse{Tv}}, Cint, Ptr{SuiteSparse_long}, Ptr{SuiteSparse_long}, Csize_t, Ptr{C_Sparse{Tv}}, Ptr{UInt8}),   
            A.p, values, C_NULL, C_NULL, 0, B.p, common())
    return B
end

# Update B as Sparse(A)
function Sparse!{Tv<:VTypes,Ti<:ITypes}(A::SparseMatrixCSC{Tv,Ti}, B::Sparse{Tv})
    s = unsafe_load(B.p)
    unsafe_copy!(s.x, pointer(A.nzval), length(A.nzval))
    @isok check_sparse(B)
    return B
end

##############################################################################
## 
## solve J'J \ J'y (used in Dogleg)
##
##############################################################################

type SparseCholeskySolver{Tv, Ti <: Integer, Tx} <: AbstractSolver
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    v::Tx
    sparseJ::Sparse{Tv}
    sparseJt::Sparse{Tv}
    F::Factor{Tv}
    cm::Array{UInt8, 1}
end

function AbstractSolver(nls::SparseLeastSquaresProblem,
    ::Type{Val{:dogleg}}, ::Type{Val{:cholesky}})
    colptr = deepcopy(nls.J.colptr)
    rowval = deepcopy(nls.J.rowval)
    sparseJ = Sparse(nls.J)
    sparseJt = transpose_(sparseJ, 2)
    cm = defaults(common()) 
    set_print_level(cm, 0)
    unsafe_store!(common_final_ll, 1)
    F = analyze(sparseJt, cm)
    return SparseCholeskySolver(colptr, rowval, _zeros(nls.x), sparseJ, sparseJt, F, cm)
end

function solve!(x::AbstractVector, J::SparseMatrixCSC, y::AbstractVector, A::SparseCholeskySolver)
    colptr, rowval, v, sparseJ, sparseJt, F, cm = 
    A.colptr, A.rowval, A.v, A.sparseJ, A.sparseJt, A.F, A.cm

    # check symbolic structure is the same
    if colptr != J.colptr || rowval != J.rowval
        error("The symbolic structure of the Jacobian has been changed. Either (i) rewrite g! so that it does not modify the structure of J (see Julia issue #9906) (ii) use solver = :iterative rather than solver = :cholesky")
    end

    Sparse!(J, sparseJ)
    transpose_unsym_!(sparseJ, 2, sparseJt)
    factorize_p!(sparseJt, 0, F, cm)
    Ac_mul_B!(v, J, y)
    # !! there is a memory allocation here
    copy!(x, F \ v)
    return x, 1
end