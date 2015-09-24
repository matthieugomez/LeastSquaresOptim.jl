[![Build Status](https://travis-ci.org/matthieugomez/SparseFactorModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/SparseFactorModels.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/SparseFactorModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/matthieugomez/SparseFactorModels.jl?branch=master)

## Motivation

This package solves least square problems. In particular, it handles problems where the Jacobian is sparse. 


Least square optimization methods require to solve `A'A x \ A' f(x)` at each step, where  A is the jacobian in dogleg, and the contatenation of the jacobian and dia(J'J) in levenberg_marquardt, 

When A is sparse, this is solved by conjugate gradient (more precisely [LSMR](http://web.stanford.edu/group/SOL/software/lsmr/) with Jacobi preconditioner).


## Syntax

```julia
ls_optim!(x, fcur, f!, J, g!; 
                method = :dogleg,
                xtol::Number = 1e-32, ftol::Number = 1e-32, grtol::Number = 1e-8,
                iterations::Integer = 100, store_trace::Bool = false)
 ```

* `x` is the initial vector
* `fcur` is an allocation for `f(x)`
* `f!(x, out)` should update `out` as `f(x)`
* `J` is a pre-allocation for the jacobian of `f`
* `g!(x, J)` should update `J` as the jacobian of `f` at `x`


`x` can be an `AbstractVector` or any type that supports the following operations
`norm(x)`, `sumabs2(x)`, `dot(x1, x2)`, `similar(x)`, `fill!(x, ::number)`, `copy!(x1, x2)`, `axpy!(α::Number, x1, x2)`, `map!(f, x...)`

`fcur` can be a `AbstractVector` or any type that supports the following operations
`sumabs2`, `scale!`, `similar`, `axpy!`

`J` can be a `AbstractMatrix`, a `SparseMatrixSC`, or any type that supports the following operations
`A_mul_B!(α::Number, A, x, β::Number, fcur)` updates fcur -> α Ax + βfcur
`Ac_mul_B!(α::Number, A, fcur, β::Number, x)` updates x -> α A'fcur + βx


## Options
You can alter the behavior ot the function by using the following keywords:
```
* `method`: What method should be used? Default to `dogleg`. Can be changed to `levenberg_marquardt`
* `xtol`: What is the threshold for determining convergence? Defaults to `1e-32`.
* `ftol`: What is the threshold for determining convergence? Defaults to `1e-32`.
* `grtol`: What is the threshold for determining convergence? Defaults to `1e-8`.
* `iterations`: How many iterations will run before the algorithm gives up? Defaults to `1_000`.
* `store_trace`: Should a trace of the optimization algorithm's state be stored? Defaults to `false`.
```
