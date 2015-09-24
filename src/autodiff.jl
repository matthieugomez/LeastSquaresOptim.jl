 using DualNumbers

function autodiff{T <: Real}(f!::Function,
                              x::Vector{T},
                              J::Matrix{T},
                              dual_in,
                              dual_out)
    @inbounds for i in 1:length(x)
        dual_in[i] = Dual(x[i], zero(T))
    end
    @inbounds for i in 1:length(x)
        dual_in[i] = Dual(x[i], one(T))
        f!(dual_in,dual_out)
        for k in 1:length(dual_out)
            J[k,i] = epsilon(dual_out[k])
        end
        dual_in[i] = Dual(real(dual_in[i]), zero(T))
    end
 end

function LeastSquaresProblem{T}(x::Vector{T}, y::Vector{T}, f!::Function, J::Matrix{T})
    dual_in = Array(Dual{T}, length(x))
    dual_out = Array(Dual{T}, length(y))
    function g!(x, J)
        autodiff(f!, x, J, dual_in, dual_out)
    end
    LeastSquaresProblem(x, y, f!, J, g!)
end