[![LeastSquaresOptim](http://pkg.julialang.org/badges/LeastSquaresOptim_0.4.svg)](http://pkg.julialang.org/?pkg=LeastSquaresOptim)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/LeastSquaresOptim.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/matthieugomez/LeastSquaresOptim.jl?branch=master)
[![Build Status](https://travis-ci.org/matthieugomez/LeastSquaresOptim.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/LeastSquaresOptim.jl)
## Motivation

This package solves non linear least squares optimization problems. The package is inspired by the [Ceres library](http://ceres-solver.org/solving.html). 

To install the package,
```julia
Pkg.add("LeastSquaresOptim")
```

## Methods

The main `optimize!` method accepts two main options : `method` and `solver`

1. `method` corresponds to a least square optimization methods:

	- `method = :levenberg_marquardt`
	- `method = :dogleg`

2. `solver` corresponds to a least squares solver. Least square optimization methods proceed by solving successively linear least squares problems `min||Ax - b||^2`. Available solvers are:

	- `solver = :qr`. Available for dense matrices
	- `solver = :cholesky`. Available for dense matrices and sparse matrices. For sparse matrices, a symbolic factorization is computed at the first iteration from SuiteSparse and numerically updated at each iteration.
	- `solve = :iterative`. A conjugate gradient method (more precisely [LSMR]([http://web.stanford.edu/group/SOL/software/lsmr/) with diagonal preconditioner). A custom type for the jacobian `A` may be specified. The following interface is expected to be defined on `A`:
		- `A_mul_B!(α::Number, A, x, β::Number, y)` updates y to αAx + βy
		- `Ac_mul_B!(α::Number, A, y, β::Number, x)` updates x to αA'y + βx
		- `colsumabs2!(x, A)` updates x to the sum of squared elements of each column
		- `size(A, d)` returns the nominal dimensions along the dth axis in the equivalent matrix representation of A.
		- `eltype(A)` returns the element type implicit in the equivalent matrix representation of A.

		Similarly, `x` or `f(x)` may be custom types. An example of the interface to define can be found in the package [SparseFactorModels.jl](https://github.com/matthieugomez/SparseFactorModels.jl).


For dense Jacobians, the default options are `method = :dogleg` and `solver = :qr`. For sparse Jacobians, the default options are  `method = :levenberg_marquardt` and `solver = :iterative`. Th `methods` and `solvers` are presented in more depth in the [Ceres documentation](http://ceres-solver.org/solving.html). 

`optimize!` also accept the options : `ftol`, `xtol`, `gr_tol`, `iterations` and `Δ` (initial radius).

## Syntax

To find `x` that minimizes `f'(x)f(x)`, construct a `LeastSquaresProblem` object with:
 - `x` is an initial set of parameters.
 - `f!` a callable object such that `f!(x, out)` writes `f(x)` in `out`.
 - `output_length` the length of the output vector. 
 And optionally
 - `g!` a function such that `g!(x, out)` writes the jacobian at x in `out`. Otherwise, the jacobian will be computed with `ForwardDiff.jl` package
 - `y` a preallocation for `f`
 - `J` a preallocation for the jacobian


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
rosenbrock_problem = LeastSquaresProblem(x = x, f! = rosenbrock_f!, output_length = 2)
optimize!(rosenbrock_problem)
```

## Memory 
The package has a particular emphasis on high dimensional problems. In particular, objects are updated in place at each method iteration: memory is allocated once and for all at the beginning of the function. 

You can avoid any initial allocation by directly passing a `LeastSquaresProblemAllocated` to the `optimize!` function. Such an object bundles a `LeastSquaresProblem` object with a few storage objects. This allows to repeteadly solve a non linear least square problems with minimal memory allocatin.
```julia
rosenbrock = LeastSquaresProblemAllocated(x, fcur, rosenbrock_f!, J, rosenbrock_g!; 
                                          method = :dogleg, solver = :qr)
optimize!(rosenbrock)
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

