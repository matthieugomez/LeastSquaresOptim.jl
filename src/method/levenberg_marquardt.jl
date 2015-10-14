
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
function AbstractMethod(nls::LeastSquaresProblem, ::Type{Val{:levenberg_marquardt}})
   LevenbergMarquardt(_zeros(nls.x), _zeros(nls.x), _zeros(nls.y), _zeros(nls.y))
end


##############################################################################
## 
## Method for LevenbergMarquardt
##
##############################################################################

const MAX_Δ = 1e16 # minimum trust region radius
const MIN_Δ = 1e-16 # maximum trust region radius
const MIN_STEP_QUALITY = 1e-3
const GOOD_STEP_QUALITY = 0.75
const MIN_DIAGONAL = 1e-6
const MAX_DIAGONAL = 1e32

function optimize!{T, Tmethod <: LevenbergMarquardt, Tsolve}(
        anls::LeastSquaresProblemAllocated{T, Tmethod, Tsolve};
            xtol::Number = 1e-8, ftol::Number = 1e-8, grtol::Number = 1e-8,
            iterations::Integer = 1_000, Δ::Number = 10.0)

    δx, dtd = anls.method.δx, anls.method.dtd
    ftrial, fpredict = anls.method.ftrial, anls.method.fpredict
    x, fcur, f!, J, g! = anls.nls.x, anls.nls.y, anls.nls.f!, anls.nls.J, anls.nls.g!

    # check
    length(x) == size(J, 2) || throw(DimensionMismatch("length(x) must equal size(J, 2)."))
    length(fcur) == size(J, 1) || throw(DimensionMismatch("length(fcur) must equal size(J, 1)."))
    length(x) == length(dtd) || throw(DimensionMismatch("The lengths of x and dtd must match."))
    length(fcur) == length(ftrial) || throw(DimensionMismatch("The lengths of fcur and ftrial must match."))
    length(ftrial) == length(fpredict) || throw(DimensionMismatch("The lengths of ftrial and fpredict must match."))

    decrease_factor = 2.0
    # initialize
    Tx, Ty = eltype(x), eltype(ftrial)
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
        clamp!(dtd, MIN_DIAGONAL, MAX_DIAGONAL)
        scale!(dtd, 1/Δ)        
        δx, lmiter = A_ldiv_B!(δx, J, fcur, dtd,  anls.solver)
        mul_calls += lmiter

        #update x
        axpy!(-one(Tx), δx, x)
        f!(x, ftrial)
        f_calls += 1

        # trial ssr
        trial_ssr = sumabs2(ftrial)

        # predicted ssr
        A_mul_B!(one(Tx), J, δx, zero(Tx), fpredict)
        mul_calls += 1
        axpy!(-one(Ty), fcur, fpredict)
        predicted_ssr = sumabs2(fpredict)

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
            Δ = min(Δ / max(1/3, 1.0 - (2.0 * ρ - 1.0)^3), MAX_Δ)
            decrease_factor = 2.0
            need_jacobian = true
        else
            # revert update
            axpy!(one(Tx), δx, x)
            Δ = max(Δ / decrease_factor , MIN_Δ)
            decrease_factor *= 2.0
        end
    end
    LeastSquaresResult("levenberg_marquardt", x, ssr, iter, converged,
                        x_converged, xtol, f_converged, ftol, gr_converged, grtol, 
                        f_calls, g_calls, mul_calls)
end


