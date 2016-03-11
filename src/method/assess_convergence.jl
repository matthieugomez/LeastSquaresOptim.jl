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

# From NLSolve
type IsFiniteException <: Exception
  indices::Vector{Int}
end
show(io::IO, e::IsFiniteException) = print(io,
  "During the resolution of the non-linear system, the evaluation" *
  " of the following equation(s) resulted in a non-finite number: $(e.indices)")

function check_isfinite(x::Vector)
    i = find(!isfinite(x))
    if !isempty(i)
        throw(IsFiniteException(i))
    end
end

