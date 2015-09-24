[![Build Status](https://travis-ci.org/matthieugomez/LeastSquaresOptim.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/LeastSquaresOptim.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/LeastSquaresOptim.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/matthieugomez/LeastSquaresOptim.jl?branch=master)
## Motivation

This package solves non linear least squares optimization problems with minimal memory footprint. The user provided Jacobian matrix can be dense (type `StridedVecOrMat`), sparse (type `SparseMatrixCSC`), or represented by operators (`A_mul_B` and `Ac_mul_B`)

1. There are two least square optimization methods

	- `method = :levenberg_marquardt`
	- `method = :dogleg`

	Either method proceeds by successive linear least squares problems `min||Ax - b||^2`, where A is a modified Jacobian at the current set of parameters.

2. The least squares problem encountered at each iteration can be solved in two different ways:

	- `solver = :factorization`. This solver can be used when the jacobian is a dense or a sparse matrix. 
		- For dense jacobians, it relies on the QR factorization in LAPACK.
		- For sparse jacobians, it relies on the cholesky factorization in SuiteSparse. A symbolic factorization is computed at the first iteration and numerically updated at each iteration. With this method, you need to make sure that `g!` does not change the symbolic structure of the jacobian `J` (i.e. access individual elements via `rowvals(J)`, not `J[i, j]`)
	- `solve = :iterative`. This solver can be used when the jacobian is a dense or sparse matrix, or any object with the following methods:
		- `A_mul_B!(α::Number, A, x, β::Number, fcur)` updates `fcur -> α Ax + βfcur`
		- `Ac_mul_B!(α::Number, A, fcur, β::Number, x)` updates `x -> α A'fcur + βx`
		- `colsumabs2!(x, J)` updates `x -> diag(J'J)`

		This solver relies on a conjugate gradient method (more precisely [LSMR]([http://web.stanford.edu/group/SOL/software/lsmr/) with diagonal preconditioner).

If the jacobian is dense, `solver` defaults to `:factorization`. and `method` defaults to `:dogleg`.Otherwise `solver` defaults to `:iterative` and `method` defaults to `levenberg_marquardt`.


## Syntax
The arguments for `NonLinearLeastSquares` are
 - `x` is an initial set of parameters
 - `fcur` is a pre-allocation for `f(x)`
 - `f!(x, out)` should update `out` as `f(x)`
 - `J` is a pre-allocation for the jacobian
 - `g!(x, J)` should update `J` as the jacobian  at `x`

`x`, `fcur` and `J` are updated in place during the function

A simple example:
```julia
using LeastSquares
x = [-1.2; 1.]
fcur = Array(Float64, 2)
J = Array(Float64, 2, 2)
function f!(x, fcur)
	fcur[1] = 1 - x[1]
	fcur[2] = 10(x[2]-x[1]^2)
end
function g!(x, J)
	J[1, 1] = -1
	J[1, 2] = 0
	J[2, 1] = -20x[1]
	J[2, 2] = 10
end
optimize!(NonLinearLeastSquares(x, fcur, f!, J, g!))
```

## Automatic differentiation
Automatic differenciation can be used for dense Jacobians thanks to the `DualNumbers` package. 
Just omit the `g!` function when constructing a `NonLinearLeastSquares` object:

```julia
optimize!(NonLinearLeastSquares(x::Vector{T}, fcur::Vector{T}, f!::Function, J::Matrix{T}))
```

## Memory 

Objects are updated in place at each iteration: memory is allocated once and for all at the beginning of the function. 

You can also avoid this initial allocation by passing a  `NonLinearLeastSquaresAllocated` object to the `optimize!` function. Such an object bundles a `NonLinearLeastSquares` object with a few storage objects.


## Related packages
Related:
- [Optim.jl](https://github.com/JuliaOpt/Optim.jl) solves general optimization problems.
- [IterativeSolvers.jl](https://github.com/JuliaLang/IterativeSolvers.jl) includes several iterative solvers for linear least squares.
- [NLSolve.jl](https://github.com/EconForge/NLsolve.jl) solves non linear equations by least squares optimization.
- [SparseFactorModels.jl](https://github.com/matthieugomez/SparseFactorModels.jl) uses this package as a back end.

## References
- Nocedal, Jorge and Stephen Wright *An Inexact Levenberg-Marquardt method for Large Sparse Nonlinear Least Squares*  (1985) The Journal of the Australian Mathematical Society
- Fong, DC. and Michael Saunders. (2011) *LSMR: An Iterative Algorithm for Sparse Least-Squares Problems*.  SIAM Journal on Scientific Computing
- Agarwal, Sameer, Keir Mierle and Others. (2010) *Ceres Solver*

