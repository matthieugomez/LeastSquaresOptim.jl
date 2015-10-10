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




function wdot(x::AbstractVector, y::AbstractVector, w::AbstractVector)
    out = zero(one(eltype(x)) * one(eltype(y)) * one(eltype(w)))
    @inbounds @simd for i in 1:length(x)
        out += w[i] * x[i] * y[i]
    end
    return out
end

# can be user written
function wdot(x, y, w)
    dot(x, y, w)
end


wsumabs2(x, w) = wdot(x, x, w)
wnorm(x, w) = sqrt(wsumabs2(x, w))
