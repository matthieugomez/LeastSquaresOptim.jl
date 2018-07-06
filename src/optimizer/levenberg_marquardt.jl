
##############################################################################
## 
## Allocations for AllocatedLevenbergMarquardt
##
##############################################################################

struct AllocatedLevenbergMarquardt{Tx1, Tx2, Ty1, Ty2} <: AbstractAllocatedOptimizer
    δx::Tx1
    dtd::Tx2
    ftrial::Ty1
    fpredict::Ty2
    function AllocatedLevenbergMarquardt{Tx1, Tx2, Ty1, Ty2}(δx, dtd, ftrial, fpredict) where {Tx1, Tx2, Ty1, Ty2}
        length(δx) == length(dtd) || throw(DimensionMismatch("The lengths of δx and dtd must match."))
        length(ftrial) == length(fpredict) || throw(DimensionMismatch("The lengths of ftrial and fpredict must match."))
        new(δx, dtd, ftrial, fpredict)
    end
end

function AllocatedLevenbergMarquardt(δx::Tx1, dtd::Tx2, ftrial::Ty1, fpredict::Ty2) where {Tx1, Tx2, Ty1, Ty2}
    AllocatedLevenbergMarquardt{Tx1, Tx2, Ty1, Ty2}(δx, dtd, ftrial, fpredict)
end


function AbstractAllocatedOptimizer(nls::LeastSquaresProblem, optimizer::LevenbergMarquardt)
   AllocatedLevenbergMarquardt(_zeros(nls.x), _zeros(nls.x), _zeros(nls.y), _zeros(nls.y))
end

##############################################################################
## 
## Optimizer for AllocatedLevenbergMarquardt
##
##############################################################################
##############################################################################


const MAX_Δ = 1e16 # minimum trust region radius
const MIN_Δ = 1e-16 # maximum trust region radius
const MIN_STEP_QUALITY = 1e-3
const GOOD_STEP_QUALITY = 0.75
const MIN_DIAGONAL = 1e-6
const MAX_DIAGONAL = 1e32

function optimize!(
    anls::LeastSquaresProblemAllocated{Tx, Ty, Tf, TJ, Tg, Toptimizer, Tsolver};
            x_tol::Number = 1e-8, f_tol::Number = 1e-8, g_tol::Number = 1e-8,
            iterations::Integer = 1_000, Δ::Number = 10.0, store_trace = false, show_trace = false, show_every = 1) where {Tx, Ty, Tf, TJ, Tg, Toptimizer <: AllocatedLevenbergMarquardt, Tsolver}

    δx, dtd = anls.optimizer.δx, anls.optimizer.dtd
    ftrial, fpredict = anls.optimizer.ftrial, anls.optimizer.fpredict
    x, fcur, f!, J, g! = anls.x, anls.y, anls.f!, anls.J, anls.g!

    decrease_factor = 2.0
    # initialize
    f_calls,  g_calls, mul_calls = 0, 0, 0
    converged, x_converged, f_converged, g_converged, converged =
        false, false, false, false, false
    f!(fcur, x)
    f_calls += 1
    ssr = sum(abs2, fcur)
    maxabs_gr = Inf
    need_jacobian = true

    eTx, eTy = eltype(x), eltype(fcur)

    iter = 0

    tr = OptimizationTrace()
    tracing = store_trace || show_trace
    tracing && update!(tr, iter, ssr, maxabs_gr, store_trace, show_trace, show_every)

    while !converged && iter < iterations 
        iter += 1
        check_isfinite(x)

        # compute step
        if need_jacobian
            g!(J, x)
            g_calls += 1
            need_jacobian = false
        end
        colsumabs2!(dtd, J)
        clamp!(dtd, MIN_DIAGONAL, MAX_DIAGONAL)
        rmul!(dtd, 1/Δ)        
        δx, lmiter = ldiv!(δx, J, fcur, dtd,  anls.solver)
        mul_calls += lmiter
        #update x
        axpy!(-one(eTx), δx, x)
        f!(ftrial, x)
        f_calls += 1

        # trial ssr
        trial_ssr = sum(abs2, ftrial)

        # predicted ssr
        mul!(fpredict, J, δx, one(eTx), zero(eTx))
        mul_calls += 1
        axpy!(-one(eTy), fcur, fpredict)
        predicted_ssr = sum(abs2, fpredict)
        ρ = (ssr - trial_ssr) / (ssr - predicted_ssr)
        mul!(dtd, J', fcur, one(eTx), zero(eTx))
        maxabs_gr = maximum(abs, dtd)
        mul_calls += 1


        x_converged, f_converged, g_converged, converged =
            assess_convergence(δx, x, maxabs_gr, ssr, trial_ssr, x_tol, f_tol, g_tol)

        if ρ > MIN_STEP_QUALITY
            copyto!(fcur, ftrial)
            ssr = trial_ssr
            # increase trust region radius (from Ceres solver)
            Δ = min(Δ / max(1/3, 1.0 - (2.0 * ρ - 1.0)^3), MAX_Δ)
            decrease_factor = 2.0
            need_jacobian = true
        else
            # revert update
            axpy!(one(eTx), δx, x)
            Δ = max(Δ / decrease_factor , MIN_Δ)
            decrease_factor *= 2.0
        end
        tracing && update!(tr, iter, ssr, maxabs_gr, store_trace, show_trace, show_every)
    end
    LeastSquaresResult("LevenbergMarquardt", x, ssr, iter, converged,
                        x_converged, x_tol, f_converged, f_tol, g_converged, g_tol, tr,
                        f_calls, g_calls, mul_calls)
end
