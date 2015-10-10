

function update!(nls::LeastSquaresProblem, δx, ftrial, fpredict)

    Tx, Ty = eltype(δx), eltype(ftrial)
    #update x
    axpy!(-one(Tx), δx, nls.x)

    # compute f at this new x
    nls.f!(nls.x, ftrial)

    # trial ssr
    trial_ssr = sumabs2(ftrial)

    # predicted ssr
    A_mul_B!(one(Tx), nls.J, δx, zero(Tx), fpredict)
    axpy!(-one(Ty), nls.y, fpredict)
    predicted_ssr = sumabs2(fpredict)

    return nls.x, ftrial, trial_ssr, predicted_ssr
end


function assess_convergence(δx,
                            x,
                            maxabs_gr,
                            ssr,
                            trial_ssr,
                            xtol::Real,
                            ftol::Real,
                            grtol::Real)


    x_converged, f_converged, gr_converged = false, false, false
    if abs(trial_ssr - ssr) <= ftol * (abs(ssr) + ftol) 
        f_converged = true
    elseif maxabs(δx) <= xtol * maxabs(x)
            x_converged = true
    elseif maxabs_gr <= grtol * maxabs(x)
            gr_converged = true
    end
    converged = x_converged || f_converged || gr_converged
    return x_converged, f_converged, gr_converged, converged
end


