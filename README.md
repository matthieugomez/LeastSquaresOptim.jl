[![LeastSquaresOptim](http://pkg.julialang.org/badges/LeastSquaresOptim_0.4.svg)](http://pkg.julialang.org/?pkg=LeastSquaresOptim)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/LeastSquaresOptim.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/matthieugomez/LeastSquaresOptim.jl?branch=master)
[![Build Status](https://travis-ci.org/matthieugomez/LeastSquaresOptim.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/LeastSquaresOptim.jl)
## Motivation

This package solves non linear least squares optimization problems. The jacobian can be a dense matrix, a sparse mamtrix, or any linear operator (any type that implements multiplications). The package is inspired by the [Ceres library](http://ceres-solver.org/solving.html). 

To install the package,
```julia
Pkg.add("LeastSquaresOptim")
```

## Methods

The main `optimize!` method accepts two main options : `method` and `solver`

1. `method` indicates a least square optimization methods:

	- `method = :levenberg_marquardt`
	- `method = :dogleg`

2. Either method proceeds by successive linear least squares problems `min||Ax - b||^2`, where A is a modified Jacobian at the current set of parameters. `solver` indicates a least squares solver for the problem encountered at each iteration:

	- `solver = :qr`. Available for dense matrices
	- `solver = :cholesky`. Available for dense matrices and sparse matrices. For sparse matrices, a symbolic factorization is computed at the first iteration from SuiteSparse and numerically updated at each iteration.
	- `solve = :iterative` corresponds to a conjugate gradient method (more precisely [LSMR]([http://web.stanford.edu/group/SOL/software/lsmr/) with diagonal preconditioner). A custom type for the jacobian `A` may be specified. The following interface is expected to be defined on `A`:
		- `A_mul_B!(α::Number, A, x, β::Number, y)` updates y to αAx + βy
		- `Ac_mul_B!(α::Number, A, y, β::Number, x)` updates x to αA'y + βx
		- `colsumabs2!(x, A)` updates x to the sum of squared elements of each column
		- `size(A, d)` returns the nominal dimensions along the dth axis in the equivalent matrix representation of A.
		- `eltype(A)` returns the element type implicit in the equivalent matrix representation of A.

		Similarly, `x` or `f(x)` don't need to be AbstractVectors. An example of the inteface to define can be found in the package [SparseFactorModels.jl](https://github.com/matthieugomez/SparseFactorModels.jl).

These different `methods` and `solvers` are presented in more depth in the [Ceres documentation](http://ceres-solver.org/solving.html). 

For dense Jacobians, default otpions are `method = :dogleg` and `solver = :qr`. For sparse Jacobians, default options are  `method = :levenberg_marquardt` and `solver = :iterative` 


## Syntax

To find `x` that minimizes `f'(x)f(x)`, construct a `LeastSquaresProblem` object with:
 - `x` is an initial set of parameters.
 - `f!` a callable object such that `f!(x, out)` writes `f(x)` in `out`.
 - `output_length` the length of the output vector. 
 - (optionally) `g!` a function such that `g!(x, out)` writes the jacobian at x in `out`. Otherwise, the jacobian will be computed with `ForwardDiff.jl` package
 - (optionally) `y` a preallocation for `f`
 - (optionally) `J` a preallocation for the jacobian


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

For all methods and solvers, `optimize!` accept the options : `method`, `solver`, `ftol`, `xtol`, `gr_tol`, `iterations` and `Δ` (initial radius).



You actually just need to specify `x`, `f!`, and `output_length`. 
When calling `optimize!`, `x`, `fcur` and `J` are updated in place during the function


## Memory 
The package has a particular emphasis on high dimensional problems. In particular, objects are updated in place at each method iteration: memory is allocated once and for all at the beginning of the function. 

You can even avoid any initial allocation by passing a `LeastSquaresProblemAllocated` to the `optimize!` function. Such an object bundles a `LeastSquaresProblem` object with a few storage objects. Since the set of storage objects depends on the method and solver used, these options should be passed to the constructor rather than the `optimize!` function:
```julia
rosenbrock = LeastSquaresProblemAllocated(x, fcur, rosenbrock_f!, J, rosenbrock_g!; 
                                          method = :dogleg, solver = :qr)
optimize!(rosenbrock)
```

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

