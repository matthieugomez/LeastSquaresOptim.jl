###############################################################################
##
## Asses convergence
##
##############################################################################

function assess_convergence(δx,
                            x,
                            maxabs_gr,
                            ssr,
                            trial_ssr,
                            xtol::Real,
                            ftol::Real,
                            grtol::Real)


    x_converged, f_converged, gr_converged = false, false, false
    maxabs_x = maxabs(x)
    if abs(trial_ssr - ssr) <= ftol * (abs(ssr) + ftol) 
        f_converged = true
    elseif maxabs(δx) <= xtol
            x_converged = true
    elseif maxabs_gr <= grtol
            gr_converged = true
    end
    converged = x_converged || f_converged || gr_converged
    return x_converged, f_converged, gr_converged, converged
end

###############################################################################
##
## Finite Exception
##
##############################################################################

type IsFiniteException <: Exception
  indices::Vector{Int}
end
Base.show(io::IO, e::IsFiniteException) = print(io,
  "During the resolution of the non-linear system, the evaluation" *
  " of the following equation(s) resulted in a non-finite number: $(e.indices)")

function check_isfinite(x::Vector)
    i = find(!isfinite(x))
    if !isempty(i)
        throw(IsFiniteException(i))
    end
end

function check_isfinite(x)
end

###############################################################################
##
## Trace
##
##############################################################################

immutable OptimizationState
    iteration::Int
    value::Float64
    g_norm::Float64
end

immutable OptimizationTrace
    states::Vector{OptimizationState}
end

OptimizationTrace() = OptimizationTrace(Vector{OptimizationState}[])
function update!(tr::OptimizationTrace,
                 iteration::Integer,
                 f_x::Real,
                 grnorm::Real,
                 store_trace::Bool,
                 show_trace::Bool,
                 show_every::Int = 1)
    os = OptimizationState(iteration, f_x, grnorm)
    if store_trace
        push!(tr, os)
    end
    if show_trace
        if iteration % show_every == 0
            show(os)
        end
    end
    return
end

function Base.show(io::IO, t::OptimizationTrace)
    @printf io "Iter     Function value   Gradient norm \n"
    @printf io "------   --------------   --------------\n"
    for state in t.states
        show(io, state)
    end
    return
end

function Base.show(io::IO, t::OptimizationState)
    @printf io "%6d   %14e   %14e\n" t.iteration t.value t.g_norm
    return
end

Base.push!(t::OptimizationTrace, s::OptimizationState) = push!(t.states, s)
Base.getindex(t::OptimizationTrace, i::Integer) = getindex(t.states, i)
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
        v[j] = sumabs2(view(A, :, j))
    end
end

function colsumabs2!(v::AbstractVector, A::SparseMatrixCSC)
    length(v) == size(A, 2) || error("v should have length size(A, 2)")
    @inbounds for j in 1:length(v)
        v[j] = sumabs2(view(nonzeros(A), nzrange(A, j)))
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
