using LeastSquaresOptim

function rosenbrock(x)
    [1 - x[1], 100 * (x[2]-x[1]^2)]
end
x0 = zeros(2)
optimize(rosenbrock, x0, Dogleg(), lower = fill(0.0, length(x0)))
