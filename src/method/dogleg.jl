##############################################################################
## 
## Allocations for Dogleg
##
##############################################################################

type Dogleg{Tx1, Tx2, Tx3, Tx4, Ty1, Ty2} <: AbstractMethod
    δgn::Tx1
    δgr::Tx2
    δx::Tx3
    dtd::Tx4
    ftrial::Ty1
    fpredict::Ty2
end

function AbstractMethod(nls::LeastSquaresProblem, ::Type{Val{:dogleg}})
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
const MIN_DIAGONAL = 1e-6
const MAX_DIAGONAL = 1e32
const DECREASE_THRESHOLD = 0.25
const INCREASE_THRESHOLD = 0.75


function optimize!{T, Tmethod <: Dogleg, Tsolve}(
    anls::LeastSquaresProblemAllocated{T, Tmethod, Tsolve};
    xtol::Number = 1e-8, ftol::Number = 1e-8, grtol::Number = 1e-8,
    iterations::Integer = 1_000, Δ::Number = 1.0)
 
     δgn, δgr, δx, dtd = anls.method.δgn, anls.method.δgr, anls.method.δx, anls.method.dtd
     ftrial, fpredict = anls.method.ftrial, anls.method.fpredict
     x, fcur, f!, J, g! = anls.nls.x, anls.nls.y, anls.nls.f!, anls.nls.J, anls.nls.g!


    # check
    length(x) == size(J, 2) || throw(DimensionMismatch("length(x) must equal size(J, 2)."))
    length(fcur) == size(J, 1) || throw(DimensionMismatch("length(fcur) must equal size(J, 1)."))
    length(x) == length(δgn) || throw(DimensionMismatch("The lengths of x and δgn must match."))
    length(x) == length(δgr) || throw(DimensionMismatch("The lengths of x and δgr must match."))
    length(x) == length(δx) || throw(DimensionMismatch("The lengths of x and δx must match."))
    length(x) == length(dtd) || throw(DimensionMismatch("The lengths of x and dtd must match."))
    length(fcur) == length(ftrial) || throw(DimensionMismatch("The lengths of fcur and ftrial must match."))
    length(ftrial) == length(fpredict) || throw(DimensionMismatch("The lengths of ftrial and fpredict must match."))

    # initialize
    Tx, Ty = eltype(x), eltype(fcur)
    reuse = false
    f_calls,  g_calls, mul_calls = 0, 0, 0
    converged, x_converged, f_converged, gr_converged, converged =
        false, false, false, false, false
    f!(x, fcur)
    f_calls += 1
    ssr = sumabs2(fcur)
    maxabs_gr = Inf

    iter = 0  
    while !converged && iter < iterations 
        iter += 1
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
            Ac_mul_B!(one(Tx), J, fcur, zero(Tx), δgr)
            mul_calls += 1
            maxabs_gr = maxabs(δgr)
            wnorm_δgr = wnorm(δgr, dtd)

            # compute Cauchy point
            map!((x, y) -> x * sqrt(y), δgn, δgr, dtd)
            A_mul_B!(one(Ty), J, δgn, zero(Ty), fpredict)
            mul_calls += 1
            α = wnorm_δgr^2 / sumabs2(fpredict)

            # compute Gauss Newton step δgn
            fill!(δgn, zero(Tx))
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
            axpy!(one(Tx), δx, x)
        end

        if ρ < DECREASE_THRESHOLD
           Δ = max(MIN_Δ, Δ * 0.5)
        elseif ρ > INCREASE_THRESHOLD
           Δ = max(Δ, 3.0 * wnorm_δx)
       end          
    end
    LeastSquaresResult("dogleg", x, ssr, iter, converged,
                        x_converged, xtol, f_converged, ftol, gr_converged, grtol, 
                        f_calls, g_calls, mul_calls)
end



