##############################################################################
##
## Methods for sparse and dense matrices
##
##############################################################################


for (name, symbol) in ((:Ac_mul_B!, 'T'),
                       (:A_mul_B!, 'N'))
    @eval begin
        $name(α::Number, A::StridedVecOrMat, x::AbstractVector, β, y::AbstractVector) = BLAS.gemm!($symbol, 'N', convert(eltype(y), α), A, x, convert(eltype(y), β), y)
    end
end

function colsumabs2!(v::AbstractVector, A::StridedVecOrMat)
    length(v) == size(A, 2) || error("v should have length size(A, 2)")
    @inbounds for j in 1:length(v)
        v[j] = sumabs2(slice(A, :, j))
    end
end

function colsumabs2!(v::AbstractVector, A::Base.SparseMatrix.SparseMatrixCSC)
    length(v) == size(A, 2) || error("v should have length size(A, 2)")
    @inbounds for j in 1:length(v)
        v[j] = sumabs2(sub(nonzeros(A), nzrange(A, j)))
    end
end

_zeros(x) = fill!(similar(x), 0)

function wdot(x::AbstractVector, y::AbstractVector, w::AbstractVector)
    (length(x) == length(y) && length(y) == length(w)) || error("vectors have not the same length")
    out = zero(one(eltype(x)) * one(eltype(y)) * one(eltype(w)))
    @inbounds for i in 1:length(x)
        out += w[i] * x[i] * y[i]
    end
    return out
end

# can be user written
wdot(x, y, w) = dot(x, y, w)
wsumabs2(x, w) = wdot(x, x, w)
wnorm(x, w) = sqrt(wsumabs2(x, w))
