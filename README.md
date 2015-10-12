[![Build Status](https://travis-ci.org/matthieugomez/LeastSquaresOptim.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/LeastSquaresOptim.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/LeastSquaresOptim.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/matthieugomez/LeastSquaresOptim.jl?branch=master)
## Motivation

This package solves non linear least squares optimization problems, with a particular emphasis on high dimensional problems:
- All operations are done in place
- The Jacobian can be a dense matrix, a sparse matrix (i.e. of type `SparseMatrixCSC`), or any object that implements multiplication operators (`A_mul_B!` and `Ac_mul_B!`).


To install the package,
```julia
Pkg.add("LeastSquaresOptim")
```

## Syntax

To find `x` that minimizes `f'(x)f(x)`, construct a `LeastSquaresProblem` object with:
 - `x` is an initial set of parameters.
 - `y` is a pre-allocation for `f(x)`.
 - `f!` a callable object such that `f!(x, out)` writes `f(x)` in `out`.
 - `J` is a pre-allocation for the jacobian.
 - `g!` a callable object such that `g!(x, out)` writes the jacobian at x in `out`.


A simple example:
```julia
using LeastSquaresOptim

function rosenbrock_f!(x, fcur)
	fcur[1] = 1 - x[1]
	fcur[2] = 100 * (x[2]-x[1]^2)
end
function rosenbrock_g!(x, J)
	J[1, 1] = -1
	J[1, 2] = 0
	J[2, 1] = -200 * x[1]
	J[2, 2] = 109
end

x = [-1.2; 1.]
fcur = Array(Float64, 2)
J = Array(Float64, 2, 2)
rosenbrock_problem = LeastSquaresProblem(x, fcur, rosenbrock_f!, J, rosenbrock_g!))
optimize!(rosenbrock_problem)
```

When calling `optimize!`, `x`, `fcur` and `J` are updated in place during the function


## Methods

The `optimize!` method accepts two main options : `method` and `solver`

1. There are two least square optimization methods

	- `method = :levenberg_marquardt`
	- `method = :dogleg`

	Either method proceeds by successive linear least squares problems `min||Ax - b||^2`, where A is a modified Jacobian at the current set of parameters.

2. The least squares problem encountered at each iteration can be solved in two different ways:

	- `solver = :qr`. Available for dense matrices
	- `solver = :cholesky`. Available for dense matrices and sparse matrices. For sparse matrices, a symbolic factorization is computed at the first iteration from SuiteSparse and numerically updated at each iteration.
	- `solve = :iterative` corresponds to a conjugate gradient method (more precisely [LSMR]([http://web.stanford.edu/group/SOL/software/lsmr/) with diagonal preconditioner). The jacobian can be a dense matrix, a sparse matrix, or any type implementing the following methods:
		- `A_mul_B!(α::Number, A, x, β::Number, fcur)`that  updates fcur -> α Ax + βfcur
		- `Ac_mul_B!(α::Number, A, fcur, β::Number, x)` that updates x -> α A'fcur + βx
		- `colsumabs2!(x, A)`, `size(A, i::Integer)` and `eltype(A)`
		
		Similarly, neither `x` or `f(x)` need to be AbstractVectors. An example can be found in the package [SparseFactorModels.jl](https://github.com/matthieugomez/SparseFactorModels.jl).

To know more about these different `methods` and `solvers`,  check the [Ceres documentation](http://ceres-solver.org/solving.html).

`optimize!` defaults depend on the type of the jacobian. 
- For dense Jacobians, defaults are `method = :dogleg` and `solver = :qr`
- For sparse Jacobians, defaults are  `method = :levenberg_marquardt` and `solver = :iterative` 


For all methods and solvers, `optimize!` also accepts the options : `ftol`, `xtol`, `gr_tol`, `iterations` and `Δ` (initial radius).

## Memory 
Objects are updated in place at each iteration: memory is allocated once and for all at the beginning of the function. 

You can even avoid any initial allocation by passing a `LeastSquaresProblemAllocated` to the `optimize!` function. Such an object bundles a `LeastSquaresProblem` object with a few storage objects. Since the set of storage objects depends on the method and solver used, these options should be passed to the constructor rather than the `optimize!` function:
```julia
rosenbrock = LeastSquaresProblemAllocated(x, fcur, rosenbrock_f!, J, rosenbrock_g!; 
                                          method = :dogleg, solver = :qr)
optimize!(rosenbrock)
```

This can be useful when alternatively minimizing a problem with respect to different parameters.

## Automatic differentiation
Automatic differenciation can be used for dense Jacobians thanks to the `ForwardDiff` package. 
Just omit the `g!` function when constructing a `LeastSquaresProblem` object:

```julia
optimize!(LeastSquaresProblem(x, fcur, rosenbrock_f!, J))
```


## Related packages
Related:
- [LSqfit.jl](https://github.com/JuliaOpt/LsqFit.jl) is a higher level package to fit curves (i.e. models of the form y = f(x, β))
- [Optim.jl](https://github.com/JuliaOpt/Optim.jl) solves general optimization problems.
- [IterativeSolvers.jl](https://github.com/JuliaLang/IterativeSolvers.jl) includes several iterative solvers for linear least squares.
- [NLSolve.jl](https://github.com/EconForge/NLsolve.jl) solves non linear equations by least squares minimization.


## References
- Nocedal, Jorge and Stephen Wright *An Inexact Levenberg-Marquardt method for Large Sparse Nonlinear Least Squares*  (1985) The Journal of the Australian Mathematical Society
- Fong, DC. and Michael Saunders. (2011) *LSMR: An Iterative Algorithm for Sparse Least-Squares Problems*.  SIAM Journal on Scientific Computing
- Agarwal, Sameer, Keir Mierle and Others. (2010) *Ceres Solver*

