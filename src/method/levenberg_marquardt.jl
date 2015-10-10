
##############################################################################
## 
## Allocations for LevenbergMarquardt
##
##############################################################################

type LevenbergMarquardt{Tx1, Tx2, Ty1, Ty2} <: AbstractMethod
    δx::Tx1
    dtd::Tx2
    ftrial::Ty1
    fpredict::Ty2
end

function allocate(nls::LeastSquaresProblem, ::Type{Val{:levenberg_marquardt}})
   LevenbergMarquardt(_zeros(nls.x), _zeros(nls.x), _zeros(nls.y), _zeros(nls.y))
end


##############################################################################
## 
## Method for LevenbergMarquardt
##
##############################################################################

const MAX_λ = 1e16 # minimum trust region radius
const MIN_λ = 1e-16 # maximum trust region radius
const MIN_STEP_QUALITY = 1e-3
const GOOD_STEP_QUALITY = 0.75
const MIN_DIAGONAL = 1e-6
const MAX_DIAGONAL = 1e32


# used directly this method to avoid any allocation within the function
function optimize!{T, Tmethod <: LevenbergMarquardt, Tsolve}(
        anls::LeastSquaresProblemAllocated{T, Tmethod , Tsolve};
            xtol::Number = 1e-8, ftol::Number = 1e-8, grtol::Number = 1e-8,
            iterations::Integer = 1_000, λ::Number = 10.0)

    decrease_factor = 2.0

    # allocations
    δx, dtd = anls.method.δx, anls.method.dtd
    ftrial, fpredict = anls.method.ftrial, anls.method.fpredict
    x, fcur, f!, J, g! = anls.nls.x, anls.nls.y, anls.nls.f!, anls.nls.J, anls.nls.g!

    # initialize
    Tx = eltype(x)
    f_calls,  g_calls, mul_calls = 0, 0, 0
    converged, x_converged, f_converged, gr_converged, converged =
        false, false, false, false, false
    f!(x, fcur)
    f_calls += 1
    ssr = sumabs2(fcur)
    need_jacobian = true

    iter = 0
    while !converged && iter < iterations 
        iter += 1

        # compute step
        if need_jacobian
            g!(x, J)
            g_calls += 1
            need_jacobian = false
        end
        colsumabs2!(dtd, J)        
        δx, lmiter = solve!(δx, dtd, λ, anls.nls, anls.solve)
        mul_calls += lmiter

        # update
        x, ftrial, trial_ssr, predicted_ssr = update!(anls.nls, δx, ftrial, fpredict)
        f_calls += 1
        mul_calls += 1
        ρ = (ssr - trial_ssr) / (ssr - predicted_ssr)

        Ac_mul_B!(one(Tx), J, fcur, zero(Tx), dtd)
        maxabs_gr = maxabs(dtd)
        mul_calls += 1

        x_converged, f_converged, gr_converged, converged =
          assess_convergence(δx, x, maxabs_gr, ssr, trial_ssr, xtol, ftol, grtol)

        if ρ > MIN_STEP_QUALITY
            copy!(fcur, ftrial)
            ssr = trial_ssr
            # increase trust region radius (from Ceres solver)
            λ = max(λ * max(1/3, 1.0 - (2.0 * ρ - 1.0)^3), MIN_λ)
            decrease_factor = 2.0
            need_jacobian = true
        else
            # revert update
            axpy!(one(Tx), δx, x)
            λ = min(λ * decrease_factor , MAX_λ)
            decrease_factor *= 2.0
        end
    end
    LeastSquaresResult(:levenberg_marquardt, x, ssr, iter, converged,
                        x_converged, xtol, f_converged, ftol, gr_converged, grtol, 
                        f_calls, g_calls, mul_calls)
end


