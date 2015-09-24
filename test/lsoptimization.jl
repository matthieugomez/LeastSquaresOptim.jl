
using LeastSquares, Base.Test

for method in [:levenberg_marquardt, :dogleg]
  function f!(x, fcur)
    fcur[1] = x[1]
    fcur[2] = 2.0 - x[2]
  end
  function g!(x, J)
    J[1, 1] = 1.0 
    J[1, 2] = 0.0
    J[2, 1]= 0.0 
    J[2, 2] = -1.0
  end
  x = [100.0, 100.0]
  fcur = Array(Float64, 2)
  J = Array(Float64, 2, 2)
  J = sprand(2, 2, 0.2)
  ls_optim(x, fcur, f!, J, g!; method = method)
  @assert norm(x - [0.0, 2.0]) < 0.01

  x = [100.0, 100.0]
  J = sprand(2, 2, 0.2)
  f(x, fcur, f!, J, g!)
  @assert norm(x - [0.0, 2.0]) < 0.01

  function f!(x, fcur)
    fcur[1] = 10.0 * (x[2] - x[1]^2 )
    fcur[2] = 1.0 - x[1]
    return fcur
  end
  function g!(x, J)
    J[1, 1] = - 20.0 * x[1]
    J[1, 2] = 10.0
    J[2, 1] = - 1.0
    J[2, 2] = 0.0
    return J
  end
  x = zeros(2)
  fcur = Array(Float64, 2)
  J = Array(Float64, 2, 2)
  ls_optim(x, fcur, f!, J, g!; method = method)
  @assert norm(x - [1.0, 1.0]) < 0.01

  x = zeros(2)
  J = sprand(2, 2, 0.2)
  ls_optim(x, fcur, f!, J, g!; method = method)
  @assert norm(x - [1.0, 1.0]) < 0.01
end