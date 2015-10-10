##############################################################################
## 
## solve J'J \ J'y by Cholesky (used in Dogleg)
##
##############################################################################

type DenseCholeskySolver{Tc} <: AbstractSolver
    chol::Tc
end

function allocate(nls::DenseLeastSquaresProblem,
    ::Type{Val{:dogleg}}, ::Type{Val{:factorization_cholesky}})
    return DenseCholeskySolver(Array(eltype(nls.J), length(nls.x), length(nls.x)))
end

function solve!(x, nls::DenseLeastSquaresProblem, solve::DenseCholeskySolver)
    y, J = nls.y, nls.J
    chol = solve.chol
    
    Ac_mul_B!(chol, J,  J)
    Ac_mul_B!(x, J,  y)
    A_ldiv_B!(cholfact!(chol, :U, Val{true}), x)
    return 1
end

##############################################################################
## 
## solve (J'J + λ dtd) \ J'y by Cholesky (used in LevenbergMarquardt)
##
##############################################################################

type DenseCholeskyDampenedSolver{Tc} <: AbstractSolver
    chol::Tc
end

function allocate(nls:: DenseLeastSquaresProblem,
    ::Type{Val{:levenberg_marquardt}}, ::Type{Val{:factorization_cholesky}})
    return DenseCholeskyDampenedSolver(Array(eltype(nls.J), length(nls.x), length(nls.x)))
end

function solve!(x, dtd, λ, nls::DenseLeastSquaresProblem, solve::DenseCholeskyDampenedSolver)
    y, J = nls.y, nls.J
    chol = solve.chol
    
    # update chol as J'J + λdtd
    Ac_mul_B!(chol, J, J)
    clamp!(dtd, MIN_DIAGONAL, MAX_DIAGONAL)
    scale!(dtd, λ)
    @inbounds @simd for i in 1:size(chol, 1)
        chol[i, i] += dtd[i]
    end

    # solve
    Ac_mul_B!(x, J, y)
    A_ldiv_B!(cholfact!(chol), x)
    return 1
end

