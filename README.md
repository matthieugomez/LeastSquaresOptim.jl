[![LeastSquaresOptim](http://pkg.julialang.org/badges/LeastSquaresOptim_0.5.svg)](http://pkg.julialang.org/?pkg=LeastSquaresOptim)
[![Build Status](https://travis-ci.org/matthieugomez/LeastSquaresOptim.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/LeastSquaresOptim.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/LeastSquaresOptim.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/matthieugomez/LeastSquaresOptim.jl?branch=master)
## Motivation

This package solves non linear least squares optimization problems. The package is inspired by the [Ceres library](http://ceres-solver.org/solving.html). 


## Syntax

To find `x` that minimizes `f'(x)f(x)`, construct a `LeastSquaresProblem` object with:
 - `x` an initial set of parameters.
 - `f`, which outputs the vector to minimize
 And optionally
 - `f!` instead of `f` to only use in place memory. In this case, also use  the option `output_length` to specify the length of the output vector. 
 - `g!` a function such that `g!(out, x)` writes the jacobian at x in `out`. Otherwise, the jacobian will be computed with the `ForwardDiff.jl` package
 - `y` a preallocation for `f`
 - `J` a preallocation for the jacobian


A simple example:
```julia
using LeastSquaresOptim

function rosenbrock_f(x)
	[1 - x[1], 100 * (x[2]-x[1]^2)]
end
x = [-1.2; 1.]
optimize!(LeastSquaresProblem(x = x, f = rosenbrock_f))

# if you want to use in place function
function rosenbrock_f!(out, x)
	out[1] = 1 - x[1]
	out[2] = 100 * (x[2]-x[1]^2)
end
optimize!(LeastSquaresProblem(x = x, f! = rosenbrock_f!, output_length = 2))

# if you want to use gradient
function rosenbrock_g!(J, x)
	J[1, 1] = -1
	J[1, 2] = 0
	J[2, 1] = -200 * x[1]
	J[2, 2] = 109
end
optimize!(LeastSquaresProblem(x = x, f = rosenbrock_f, g! = rosenbrock_g!))
```

## Optimizer and Solver

The main `optimize!` method accepts two main arguments : `optimizer` and `solver`

1. Choose an optimization method:

	- `LeastSquaresOptim.LevenbergMarquardt()`
	- `LeastSquaresOptim.Dogleg()`

2. Choose a least square solver (a least square optimization method proceeds by solving successively linear least squares problems `min||Ax - b||^2`). 
	- `LeastSquaresOptim.QR()`. Available for dense jacobians
	- `LeastSquaresOptim.Cholesky()`. Available for dense jacobians
	- `LeastSquaresOptim.LSMR()`. A conjugate gradient method ([LSMR]([http://web.stanford.edu/group/SOL/software/lsmr/) with diagonal preconditioner). The jacobian can be of any type that defines the following interface is defined:
		- `A_mul_B!(α::Number, A, x, β::Number, y)` updates y to αAx + βy
		- `Ac_mul_B!(α::Number, A, y, β::Number, x)` updates x to αA'y + βx
		- `colsumabs2!(x, A)` updates x to the sum of squared elements of each column
		- `size(A, d)` returns the nominal dimensions along the dth axis in the equivalent matrix representation of A.
		- `eltype(A)` returns the element type implicit in the equivalent matrix representation of A.

		Similarly, `x` or `f(x)` may be custom types. An example of the interface to define can be found in the package [SparseFactorModels.jl](https://github.com/matthieugomez/SparseFactorModels.jl).

		For the `LSMR` solver, you can optionally specifying a function `preconditioner!` and a matrix `P` such that `preconditioner(x, J, P)` updates `P` as a preconditioner for `J'J` in the case of a Dogleg optimization method, and such that `preconditioner(x, J, λ, P)` updates `P` as a preconditioner for `J'J + λ` in the case of LevenbergMarquardt optimization method. By default, the preconditioner is chosen as the diagonal of of the matrix `J'J`. The preconditioner can be any type that supports `A_ldiv_B!(x, P, y)`

The `optimizers` and `solvers` are presented in more depth in the [Ceres documentation](http://ceres-solver.org/solving.html). For dense jacobians, the default options are `Dogle()` and `QR()`. For sparse jacobians, the default options are  `LevenbergMarquardt` and `LSMR()`. 

`optimize!` also accept the options : `ftol`, `xtol`, `grtol`, `iterations` and `Δ` (initial radius).



## Memory 
The package is written with large scale problems in mind. In particular, memory is allocated once and for all at the start of the function call ; objects are updated in place at each method iteration. 

You can even avoid initial allocations by directly passing a `LeastSquaresProblemAllocated` to the `optimize!` function. Such an object bundles a `LeastSquaresProblem` object with a few storage objects. This may be useful when repeatedly solving non linear least square problems.
```julia
rosenbrock = LeastSquaresProblemAllocated(x, fcur, rosenbrock_f!, J, rosenbrock_g!; 
                                          LeastSquaresOptim.Dogleg(), LeastSquaresOptim.QR())
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

## Installation
To install the package,
```julia
Pkg.add("LeastSquaresOptim")
```