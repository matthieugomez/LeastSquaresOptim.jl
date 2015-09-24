##############################################################################
##
## dogleg method
##
## Reference: Is Levenberg-Marquardt the Most Efficient Optimization Algorithm for Implementing Bundle Adjustment? 
## Manolis I.A. Lourakis and Antonis A. Argyros
##
## x is any type that implements: norm, sumabs2, dot, similar, fill!, copy!, axpy!, map!
## fcur is any type that implements: sumabs2(fcur), scale!(fcur, α), similar(fcur), axpy!
## J is a matrix, a SparseMatrixSC, or anything that implements
## sumabs21(vec, J) : updates vec to sumabs(J, 1)
## A_mul_B!(α, A, b, β, c) updates c -> α Ab + βc
## Ac_mul_B!(α, A, b, β, c) updates c -> α A'b + βc
##
## f!(x, out) returns vector f_i(x) in in out
## g!(x, out) returns jacobian in out
##
## x is the initial solution and is transformed in place to the solution
##
##############################################################################
const MIN_Δ = 1e-16 # maximum trust region radius
const MAX_Δ = 1e16 # minimum trust region radius
const MIN_STEP_QUALITY = 1e-3
const GOOD_STEP_QUALITY = 0.75

function ls_optim!(::Type{Val{:dogleg}}, x, fcur, f!, J, g!; 
                xtol::Number = 1e-32, ftol::Number = 1e-32, grtol::Number = 1e-8,
                iterations::Integer = 100, Δ::Number = 1.0)
 
    # temporary array
    δgn = similar(x) # gauss newton step
    δsd = similar(x) # steepest descent
    δdiff = similar(x) # δgn - δsd
    δx = similar(x)
    ftrial = similar(fcur)
    ftmp = similar(fcur)

    # temporary arrays used in computing least square
    alloc = ls_solver_alloc(Val{:dogleg}, x, fcur, J)

    # initialize
    f_calls = 0 
    g_calls = 0
    f!(x, fcur)
    f_calls += 1
    mfcur = scale!(fcur, -1)
    ssr = sumabs2(fcur)
    iter = 0
    while !converged && iter < iterations 
        iter += 1
        g!(x, J)
        g_calls += 1
        Ac_mul_B!(δsd, J, mfcur)
        A_mul_B!(ftmp, J, δsd)
        scale!(δsd, sumabs2(δsd) / sumabs2(ftmp))
        gncomputed = false
        ρ = -1
        while ρ < MIN_STEP_QUALITY
            # compute δx
            if norm(δsd) >= Δ
                # Cauchy point is out of the region
                # take largest Cauchy step within the trust region boundary
                scale!(δx, δsd, Δ / norm(δsd))
            else
                if (!gncomputed)
                    fill!(δgn, zero(Float64))
                    ls_iter = ls_solver!(Val{:dogleg}, δgn, mfcur, J, alloc)
                    mull_calls += ls_iter
                    # δdiff = δgn - δsd
                    copy!(δdiff, δgn)
                    axpy!(-1, δsd,  δdiff)
                    gncomputed = true
                end
                if norm(δgn) <= Δ
                    # Gauss-Newton step is within the region
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
            axpy!(1, δx, x)
            f!(x, ftrial)
            f_calls += 1
            trial_ssr = sumabs2(ftrial)

            if abs(ssr - trial_ssr) <= max(tol^2 * ssr, eps()^2)
                return iter, true
            end

            if isa(J, Matrix)
                      A_mul_B!(ftmp, J, δx)
                  else
                      A_mul_B!(1.0, J, δx, 0.0, ftmp)
                  end
            axpy!(-1.0, mfcur, ftmp)
            predicted_ssr = sumabs2(ftmp)
            ρ = (trial_ssr - ssr) / (predicted_ssr - ssr)

            if ρ >= MIN_STEP_QUALITY
                # test convergence
                if isa(J, Matrix)
                    Ac_mul_B!(δx, J, fcur)
                else
                    Ac_mul_B!(1.0, J, fcur, 0.0, δx)
                end
                x_converged, f_converged, gr_converged, converged =
                    assess_convergence(δx, trial_ssr, ssr, f, δx, xtol, ftol, grtol)
                # Successful iteration
                copy!(fcur, ftrial)
                mfcur = scale!(fcur, -1)
                ssr = trial_ssr
            else
                # unsucessful iteration
                axpy!(-1.0, δx, x)
            end
            if ρ < 0.25
               Δ = max(MIN_Δ, Δ / 2)
            elseif ρ > GOOD_STEP_QUALITY
               Δ = min(MAX_Δ, 2 * Δ)
           end          
        end
    end
    return LSResults("Dogleg",
                       x,
                       ssr,
                       iter,
                       iter == iterations,
                       x_converged,
                       xtol,
                       f_converged,
                       ftol,
                       gr_converged,
                       grtol,
                       f_calls,
                       g_calls, 
                       mu_calls)
    end

##############################################################################
## 
## Case of Dense Matrix
##
##############################################################################

function ls_solver_alloc{T}(::Type{Val{:dogleg}, x::Vector{T}, mfcur::Vector{T}, J::Matrix{T})
    nothing
end

function ls_solver!{T}(::Type{Val{:dogleg}, δx::Vector{T}, mfcur::Vector{T}, J::Matrix{T}, alloc::Void)
    δx[:] = J \ mfcur
    return 1
end

##############################################################################
## 
## Case where J'J is costly to store: Sparse Matrix, or anything
## that defines two functions
## A_mul_B(α, A, a, β b) that updates b as α A a + β b 
## Ac_mul_B(α, A, a, β b) that updates b as α A' a + β b 
## sumabs21(x, A) that updates x with sumabs2(A)
##
## we use LSMR for the problem J'J \ J' fcur 
## with 1/sqrt(diag(J'J)) as preconditioner
##############################################################################

type MatrixWrapperDogleg{TA, Tx}
    A::TA
    normalization::Tx 
    tmp::Tx
end

function A_mul_B!{TA, Tx}(α::Number, mw::MatrixWrapperDogleg{TA, Tx}, a::Tx, 
                β::Number, b)
    map!((x, z) -> x * z, mw.tmp, a, mw.normalization)
    A_mul_B!(α, mw.A, mw.tmp, β, b)
    return b
end

function Ac_mul_B!{TA, Tx}(α::Number, mw::MatrixWrapperDogleg{TA, Tx}, a, 
                β::Number, b::Tx)
    Ac_mul_B!(1, mw.A, a, 0, mw.tmp)
    map!((x, z) -> x * z, mw.tmp, mw.tmp, mw.normalization)
    if β != 1
        if β == 0
            fill!(b, 0)
        else
            scale!(b, β)
        end
    end
    axpy!(α, mw.tmp, b)
    return b
end

function ls_solver_alloc(::Type{Val{:dogleg}, x, fcur, J)
    normalization = similar(x)
    tmp = similar(x)
    fill!(tmp, zero(Float64))
    u = similar(fcur)
    v = similar(x)
    h = similar(x)
    hbar = similar(x)
    return normalization, tmp, u, v, h, hbar
end

function ls_solver!(::Type{Val{:dogleg}, δx, mfcur, J, alloc)
    normalization, tmp, u, v, h, hbar = alloc
    sumabs21!(normalization, J)
    map!(x -> x == 0 ? 0 : 1 / sqrt(x), normalization, normalization)
    A = MatrixWrapperDogleg(J, normalization, tmp)
    iter = lsmr!(δx, mfcur, A, u, v, h, hbar)
    map!((x, z) -> x * z, δx, δx, normalization)
    return iter
end
