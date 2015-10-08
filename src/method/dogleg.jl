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
    ftmp::Ty2
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
    ftrial, ftmp = anls.method.ftrial, anls.method.ftmp
    x, fcur, f!, J, g! = anls.nls.x, anls.nls.y, anls.nls.f!, anls.nls.J, anls.nls.g!
    f_calls,  g_calls, mul_calls = 0, 0, 0
    x_converged, f_converged, gr_converged, converged =
        false, false, false, false

  
    Tx, Ty = eltype(x), eltype(fcur)
    f!(x, fcur)
    f_calls += 1
    ssr = sumabs2(fcur)
    iter = 0
    while !converged && iter < iterations 
        iter += 1
        g!(x, J)
        g_calls += 1
        _Ac_mul_B!(δsd, J, fcur)
        _A_mul_B!(ftmp, J, δsd)
        mul_calls += 2
        scale!(δsd, sumabs2(δsd) / sumabs2(ftmp))
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
                    ls_iter = solve!(anls)
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

            # update x
            axpy!(-one(Tx), δx, x)
            f!(x, ftrial)
            f_calls += 1
            trial_ssr = sumabs2(ftrial)

            _A_mul_B!(ftmp, J, δx)
            mul_calls += 1
            axpy!(-one(Ty), fcur, ftmp)
            predicted_ssr = sumabs2(ftmp)

            # test convergence
            _Ac_mul_B!(δdiff, J, fcur)
            mul_calls += 1
             x_converged, f_converged, gr_converged, converged =
                assess_convergence(δx, x, δdiff, trial_ssr, ssr, xtol, ftol, grtol)
            ρ = (ssr - trial_ssr) / (ssr - predicted_ssr)
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
