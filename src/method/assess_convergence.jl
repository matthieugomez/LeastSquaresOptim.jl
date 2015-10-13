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


