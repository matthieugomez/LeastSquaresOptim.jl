##############################################################################
## 
## Allocations for AllocatedDogleg
##
##############################################################################

struct AllocatedDogleg{Tx1, Tx2, Tx3, Tx4, Ty1, Ty2} <: AbstractAllocatedOptimizer
    δgn::Tx1
    δgr::Tx2
    δx::Tx3
    dtd::Tx4
    ftrial::Ty1
    fpredict::Ty2
    function AllocatedDogleg{Tx1, Tx2, Tx3, Tx4, Ty1, Ty2}(δgn, δgr, δx, dtd, ftrial, fpredict) where {Tx1, Tx2, Tx3, Tx4, Ty1, Ty2}
        length(δgn) == length(δgr) || throw(DimensionMismatch("The lengths of δgn and δgr must match."))
        length(δgn) == length(δx) || throw(DimensionMismatch("The lengths of δgn and δx must match."))
        length(δgn) == length(dtd) || throw(DimensionMismatch("The lengths of δgn and dtd must match."))
        length(ftrial) == length(fpredict) || throw(DimensionMismatch("The lengths of ftrial and fpredict must match."))
        new(δgn, δgr, δx, dtd, ftrial, fpredict)
    end
end

function AllocatedDogleg(δgn::Tx1, δgr::Tx2, δx::Tx3, dtd::Tx4, ftrial::Ty1, fpredict::Ty2) where {Tx1, Tx2, Tx3, Tx4, Ty1, Ty2}
    AllocatedDogleg{Tx1, Tx2, Tx3, Tx4, Ty1, Ty2}(δgn, δgr, δx, dtd, ftrial, fpredict)
end

function AbstractAllocatedOptimizer{Tx, Ty, Tf, TJ, Tg}(nls::LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}, optimizer::Dogleg)
    AllocatedDogleg(_zeros(nls.x), _zeros(nls.x), _zeros(nls.x), _zeros(nls.x), 
    _zeros(nls.y), _zeros(nls.y))
end

##############################################################################
## 
## Method for AllocatedDogleg
##
##############################################################################

const MIN_Δ = 1e-16 # maximum trust region radius
const MAX_Δ = 1e16 # minimum trust region radius
const MIN_STEP_QUALITY = 1e-3
const MIN_DIAGONAL = 1e-6
const MAX_DIAGONAL = 1e32
const DECREASE_THRESHOLD = 0.25
const INCREASE_THRESHOLD = 0.75

function optimize!{Tx, Ty, Tf, TJ, Tg, Toptimizer <: AllocatedDogleg, Tsolver}(
    anls::LeastSquaresProblemAllocated{Tx, Ty, Tf, TJ, Tg, Toptimizer, Tsolver};
    xtol::Number = 1e-8, ftol::Number = 1e-8, grtol::Number = 1e-8,
    iterations::Integer = 1_000, Δ::Number = 1.0, store_trace = false, show_trace = false, show_every = 1)
 
     δgn, δgr, δx, dtd = anls.optimizer.δgn, anls.optimizer.δgr, anls.optimizer.δx, anls.optimizer.dtd
     ftrial, fpredict = anls.optimizer.ftrial, anls.optimizer.fpredict
     x, fcur, f!, J, g! = anls.x, anls.y, anls.f!, anls.J, anls.g!

    # initialize
    reuse = false
    f_calls,  g_calls, mul_calls = 0, 0, 0
    converged, x_converged, f_converged, gr_converged, converged =
        false, false, false, false, false
    f!(x, fcur)
    f_calls += 1
    ssr = norm(fcur, 2)^2
    maxabs_gr = Inf

    iter = 0  
    eTx, eTy = eltype(x), eltype(fcur)

    tr = OptimizationTrace()
    tracing = store_trace || show_trace
    tracing && update!(tr, iter, ssr, maxabs_gr, store_trace, show_trace, show_every)
    while !converged && iter < iterations 
        iter += 1
        check_isfinite(x)
        # compute step
        if !reuse
            #update gradient
            g!(x, J)
            g_calls += 1
            colsumabs2!(dtd, J)
            clamp!(dtd, MIN_DIAGONAL, MAX_DIAGONAL)

            if iter == 1
                wnorm_x = wnorm(x, dtd)
                if wnorm_x > 0
                    Δ *= wnorm_x
                end
            end
            # compute (opposite) gradient
            Ac_mul_B!(one(eTx), J, fcur, zero(eTx), δgr)
            mul_calls += 1
            maxabs_gr = norm(δgr, Inf)
            wnorm_δgr = wnorm(δgr, dtd)

            # compute Cauchy point
            map!((x, y) -> x * sqrt(y), δgn, δgr, dtd)
            A_mul_B!(one(eTy), J, δgn, zero(eTy), fpredict)
            mul_calls += 1
            α = wnorm_δgr^2 / norm(fpredict, 2)^2

            # compute Gauss Newton step δgn
            fill!(δgn, zero(eTx))
            δgn, ls_iter = A_ldiv_B!(δgn, J, fcur, anls.solver)
            mul_calls += ls_iter
            wnorm_δgn = wnorm(δgn, dtd)
        end
        # compute δx
        if wnorm_δgn <= Δ
            #  Case 1. The Gauss-Newton step lies inside the trust region
            copy!(δx, δgn)
            wnorm_δx = wnorm_δgn
        elseif wnorm_δgr * α >= Δ
            # Case 2. The Cauchy point and the Gauss-Newton steps lie outside
            # the trust region.
            # rescale Cauchy step within the trust region and return
            copy!(δx, δgr)
            scale!(δx, Δ / wnorm_δgr)
            wnorm_δx = Δ
        else
            # Case 3. The Cauchy point is inside the trust region nd the
            # Gauss-Newton step is outside
            b_dot_a = α * wdot(δgr, δgn, dtd)
            a_squared_norm = (α * wnorm_δgr)^2
            b_minus_a_squared_norm =
                  a_squared_norm - 2 * b_dot_a + wnorm_δgn^2
            c =  b_dot_a - a_squared_norm
            d = sqrt(c^2 + b_minus_a_squared_norm * (Δ^2 - a_squared_norm))
            β = (c <= 0) ? (d - c)/ b_minus_a_squared_norm : (Δ^2 - a_squared_norm) / (d + c)
            copy!(δx, δgn)
            scale!(δx, β)
            axpy!(α * (1 - β), δgr, δx)
            wnorm_δx = wnorm(δx, dtd)
        end


        #update x
        axpy!(-one(eTx), δx, x)
        f!(x, ftrial)
        f_calls += 1

        # trial ssr
        trial_ssr = norm(ftrial, 2)^2

        # predicted ssr
        A_mul_B!(one(eTx), J, δx, zero(eTx), fpredict)
        mul_calls += 1
        axpy!(-one(eTy), fcur, fpredict)
        predicted_ssr = norm(fpredict, 2)^2

        ρ = (ssr - trial_ssr) / (ssr - predicted_ssr)
        x_converged, f_converged, gr_converged, converged = 
            assess_convergence(δx, x, maxabs_gr, ssr, trial_ssr, xtol, ftol, grtol)

        if ρ >= MIN_STEP_QUALITY
            # Successful iteration
            reuse = false
            copy!(fcur, ftrial)
            ssr = trial_ssr
        else
            # unsucessful iteration
            reuse = true
            axpy!(one(eTx), δx, x)
        end

        if ρ < DECREASE_THRESHOLD
           Δ = max(MIN_Δ, Δ * 0.5)
        elseif ρ > INCREASE_THRESHOLD
           Δ = max(Δ, 3.0 * wnorm_δx)
       end  
       tracing && update!(tr, iter, ssr, maxabs_gr, store_trace, show_trace, show_every) 
    end
    LeastSquaresResult("Dogleg", x, ssr, iter, converged,
                        x_converged, xtol, f_converged, ftol, gr_converged, grtol, tr, 
                        f_calls, g_calls, mul_calls)
end
