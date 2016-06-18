[![LeastSquaresOptim](http://pkg.julialang.org/badges/LeastSquaresOptim_0.4.svg)](http://pkg.julialang.org/?pkg=LeastSquaresOptim)
[![Build Status](https://travis-ci.org/matthieugomez/LeastSquaresOptim.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/LeastSquaresOptim.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/LeastSquaresOptim.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/matthieugomez/LeastSquaresOptim.jl?branch=master)
## Motivation

This package solves large non linear least squares problems. The package is inspired by the [Ceres library](http://ceres-solver.org/solving.html). 

To install the package,
```julia
Pkg.add("LeastSquaresOptim")
```

## Optimizer and Solver

The main `optimize!` method accepts two main options : `optimizer` and `solver`

1. The `optimizer` option allows to choose a particular optimization method:

	- `optimizer = :levenberg_marquardt`
	- `optimizer = :dogleg`

2. The `solver` option allows to choose a particular least square solver (a least square optimization method proceeds by solving successively linear least squares problems `min||Ax - b||^2`). 
	- `solver = :qr`. Available for dense jacobians
	- `solver = :cholesky`. Available for dense jacobians and sparse jacobians. For sparse jacobians, a symbolic factorization is computed at the first iteration from SuiteSparse and numerically updated at each iteration.
	- `solve = :iterative`. A conjugate gradient method ([LSMR]([http://web.stanford.edu/group/SOL/software/lsmr/) with diagonal preconditioner). The jacobian can be of any type that defines the following interface is defined:
		- `A_mul_B!(α::Number, A, x, β::Number, y)` updates y to αAx + βy
		- `Ac_mul_B!(α::Number, A, y, β::Number, x)` updates x to αA'y + βx
		- `colsumabs2!(x, A)` updates x to the sum of squared elements of each column
		- `size(A, d)` returns the nominal dimensions along the dth axis in the equivalent matrix representation of A.
		- `eltype(A)` returns the element type implicit in the equivalent matrix representation of A.

		Similarly, `x` or `f(x)` may be custom types. An example of the interface to define can be found in the package [SparseFactorModels.jl](https://github.com/matthieugomez/SparseFactorModels.jl).

The `optimizers` and `solvers` are presented in more depth in the [Ceres documentation](http://ceres-solver.org/solving.html). For dense jacobians, the default options are `optimizer = :dogleg` and `solver = :qr`. For sparse jacobians, the default options are  `optimizer = :levenberg_marquardt` and `solver = :iterative`. 

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
The package is written with large scale problems in mind. In particular, memory is allocated once and for all at the start of the function call ; objects are updated in place at each method iteration. 

You can even avoid initial allocations by directly passing a `LeastSquaresProblemAllocated` to the `optimize!` function. Such an object bundles a `LeastSquaresProblem` object with a few storage objects. This may be useful when repeatedly solving non linear least square problems.
```julia
rosenbrock = LeastSquaresProblemAllocated(x, fcur, rosenbrock_f!, J, rosenbrock_g!; 
                                          optimizer = :dogleg, solver = :qr)
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

