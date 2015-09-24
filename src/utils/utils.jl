##############################################################################
##
## Methods for sparse and dense matrices
##
##############################################################################

for name in (:Ac_mul_B!, :A_mul_B!)
    _name = parse("_$name")
    @eval begin
        $_name(y::AbstractVector, X::StridedVecOrMat, x::AbstractVector) = $name(y, X, x)
        $_name(y, X, x) = $name(1, X, x, 0, y)
    end
end

for (name, symbol) in ((:Ac_mul_B!, 'T'),
                       (:A_mul_B!, 'N'))
    @eval begin
        $name(α::Number, A::StridedVecOrMat, x::AbstractVector, β, y::AbstractVector) = BLAS.gemm!($symbol, 'N', convert(eltype(y), α), A, x, convert(eltype(y), β), y)
    end
end

function colsumabs2!(dtd::AbstractVector, J::StridedVecOrMat)
    for i in 1:length(dtd)
        dtd[i] = sumabs2(slice(J, :, i))
    end
end

function colsumabs2!(v::AbstractVector, A::Base.SparseMatrix.SparseMatrixCSC)
    for i in 1:length(v)
        v[i] = sumabs2(sub(nonzeros(A), nzrange(A, i)))
    end
end


_zeros(x) = fill!(similar(x), 0)
