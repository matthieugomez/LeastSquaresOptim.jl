##############################################################################
## 
## Allocations for Dogleg
##
##############################################################################

type Dogleg{Tx1, Tx2, Tx3, Tx4, Ty1, Ty2} <: AbstractMethod
    δgn::Tx1
    δsd::Tx2
    δdiff::Tx3
    δx::Tx4
    ftrial::Ty1
    fpredict::Ty2
end

function allocate(nls::LeastSquaresProblem, ::Type{Val{:dogleg}})
    Dogleg(_zeros(nls.x), _zeros(nls.x), _zeros(nls.x), _zeros(nls.x), 
    _zeros(nls.y), _zeros(nls.y))
end

##############################################################################
## 
## Method for Dogleg
##
##############################################################################

const MIN_Δ = 1e-16 # maximum trust region radius
const MAX_Δ = 1e16 # minimum trust region radius
const MIN_STEP_QUALITY = 1e-3
const DECREASE_THRESHOLD = 0.25
const INCREASE_THRESHOLD = 0.75

function optimize!{T, Tmethod <: Dogleg, Tsolve}(
    anls::LeastSquaresProblemAllocated{T, Tmethod, Tsolve};
    xtol::Number = 1e-8, ftol::Number = 1e-8, grtol::Number = 1e-8,
    iterations::Integer = 1_000, Δ::Number = 1.0)
 
    δgn, δsd, δdiff, δx = anls.method.δgn, anls.method.δsd, anls.method.δdiff, anls.method.δx
    ftrial, fpredict = anls.method.ftrial, anls.method.fpredict
    x, fcur, f!, J, g! = anls.nls.x, anls.nls.y, anls.nls.f!, anls.nls.J, anls.nls.g!
    Tx, Ty = eltype(x), eltype(fcur)

    f_calls,  g_calls, mul_calls = 0, 0, 0
    converged, x_converged, f_converged, gr_converged, converged =
        false, false, false, false, false
    f!(x, fcur)
    f_calls += 1
    ssr = sumabs2(fcur)


    iter = 0  
    while !converged && iter < iterations 
        iter += 1
        g!(x, J)
        g_calls += 1
        Ac_mul_B!(one(Tx), J, fcur, zero(Tx), δsd)
        A_mul_B!(one(Ty), J, δsd, zero(Ty), fpredict)
        maxabs_gr = maxabs(δsd)
        mul_calls += 2
        scale!(δsd, sumabs2(δsd) / sumabs2(fpredict))
        gncomputed = false
        ρ = -1
        while !converged && ρ < 0
            # compute δx
            if norm(δsd) >= Δ
                # Cauchy point is out of the region
                # rescale Cauchy step within the trust region and return
                copy!(δx, δsd)
                scale!(δx, Δ / norm(δsd))
            else
                if (!gncomputed)
                    ls_iter = solve!(δgn, anls.nls, anls.solve)
                    mul_calls += ls_iter
                    gncomputed = true
                end
                # δdiff = δgn - δsd
                copy!(δdiff, δgn)
                axpy!(-one(Tx), δsd,  δdiff)
                if norm(δgn) <= Δ
                    # Gauss-Newton step is within the region
                    # It is the optimal solution to the trust-region problem.
                    copy!(δx, δgn)
                else
                    # Gauss-Newton step is outside the region
                    # intersection trust region and line Cauchy point and the Gauss-Newton step
                    b = 2 * dot(δsd, δdiff)
                    a = sumabs2(δdiff)
                    c = sumabs2(δsd)
                    tau = (-b + sqrt(b^2 - 4 * a * (c - Δ^2)))/(2*a) 
                    copy!(δx, δsd)
                    axpy!(tau, δdiff, δx)
                end
            end

            x, ftrial, trial_ssr, predicted_ssr = update!(anls.nls, δx, ftrial, fpredict)
            f_calls += 1
            mul_calls += 1
            ρ = (ssr - trial_ssr) / (ssr - predicted_ssr)

            x_converged, f_converged, gr_converged, converged = 
                assess_convergence(δx, x, maxabs_gr, ssr, trial_ssr, xtol, ftol, grtol)

            if ρ >= MIN_STEP_QUALITY
                # Successful iteration
                copy!(fcur, ftrial)
                ssr = trial_ssr
            else
                # unsucessful iteration
                axpy!(one(Tx), δx, x)
            end
            if ρ < DECREASE_THRESHOLD
               Δ = max(MIN_Δ, Δ * 0.5)
            elseif ρ > INCREASE_THRESHOLD
               Δ = max(Δ, 3.0 * norm(δx))
           end          
        end
    end
    LeastSquaresResult(:dogleg, x, ssr, iter, converged,
                        x_converged, xtol, f_converged, ftol, gr_converged, grtol, 
                        f_calls, g_calls, mul_calls)
end
