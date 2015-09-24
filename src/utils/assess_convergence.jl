
function assess_convergence(δx,
                            x,
                            g,
                            trial_ssr,
                            ssr,
                            xtol::Real,
                            ftol::Real,
                            grtol::Real)

    x_converged, f_converged, gr_converged = false, false, false
    if abs(trial_ssr - ssr) <= ftol * (abs(ssr) + ftol) 
        f_converged = true
    elseif maxabs(δx) <= xtol * maxabs(x)
            x_converged = true
    elseif maxabs(g) <= grtol * maxabs(x)
            gr_converged = true
    end
    converged = x_converged || f_converged || gr_converged
    return x_converged, f_converged, gr_converged, converged
end
