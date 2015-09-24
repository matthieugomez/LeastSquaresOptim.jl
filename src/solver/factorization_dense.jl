
##############################################################################
## 
## Dogleg : solve J'J \ J'y
##
##############################################################################
type DoglegDense{TJ} <: AbstractSolver
    qr::TJ
end

function allocate(nls::DenseLeastSquaresProblem,
    ::Type{Val{:dogleg}}, ::Type{Val{:factorization}})
    return DoglegDense(deepcopy(nls.J))
end

function solve!{T <: DenseLeastSquaresProblem, Tmethod <: Dogleg, Tsolve <: DoglegDense}(
    anls::LeastSquaresProblemAllocated{T, Tmethod, Tsolve})
    copy!(anls.solve.qr, anls.nls.J)
    copy!(anls.method.δgn, anls.nls.y)
    A_ldiv_B!(qrfact!(anls.solve.qr), anls.method.δgn)
    return 1
end

##############################################################################
## 
## LevenbergMarquardt: solve (J'J + λ dtd) \ J'y
##
##############################################################################

type DenseLevenbergMarquardt{TM} <: AbstractSolver
    M::TM
end

function allocate(nls:: DenseLeastSquaresProblem,
    ::Type{Val{:levenberg_marquardt}}, ::Type{Val{:factorization}})
    return DenseLevenbergMarquardt(Array(eltype(nls.J), length(nls.x), length(nls.x)))
end

function solve!{T <: DenseLeastSquaresProblem, Tmethod <: LevenbergMarquardt, Tsolve <: DenseLevenbergMarquardt}(
    anls::LeastSquaresProblemAllocated{T, Tmethod, Tsolve}, λ)
    y, J = anls.nls.y, anls.nls.J
    M = anls.solve.M
    dtd, δx = anls.method.dtd, anls.method.δx
    
    # update M as J'J + λdtd
    At_mul_B!(M, J, J)
    clamp!(dtd, MIN_DIAGONAL, Inf)
    scale!(dtd, λ)
    @inbounds @simd for i in 1:size(M, 1)
        M[i, i] += dtd[i]
    end

    # update u as J' fcur
    At_mul_B!(anls.method.δx, J, y)

    # solve
    A_ldiv_B!(cholfact!(M), anls.method.δx)
    return 1
end