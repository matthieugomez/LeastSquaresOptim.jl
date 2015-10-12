# This file replicates the testsuite provided by MINPACK for the hybrj
# function. It consists in 14 functions with their analytical Jacobian.
# Julia replication taken from NLSOLVE.jl package
using LeastSquaresOptim, Base.Test


function rosenbrock()
    name = "rosenbrock"
    function f!(x, fvec)
        fvec[1] = 1 - x[1]
        fvec[2] = 10(x[2]-x[1]^2)
    end
    function g!(x, fjac)
        fjac[1,1] = -1
        fjac[1,2] = 0
        fjac[2,1] = -20x[1]
        fjac[2,2] = 10
    end
    x = [-1.2; 1.]
    return name, f!, g!, x
end

function powell_singular()
    name = "powell_singular"
    function f!(x, fvec)
        fvec[1] = x[1] + 10x[2]
        fvec[2] = sqrt(5)*(x[3] - x[4])
        fvec[3] = (x[2] - 2x[3])^2
        fvec[4] = sqrt(10)*(x[1] - x[4])^2
    end
    function g!(x, fjac)
        fill!(fjac, 0)
        fjac[1,1] = 1
        fjac[1,2] = 10
        fjac[2,3] = sqrt(5)
        fjac[2,4] = -fjac[2,3]
        fjac[3,2] = 2(x[2] - 2x[3])
        fjac[3,3] = -2fjac[3,2]
        fjac[4,1] = 2sqrt(10)*(x[1] - x[4])
        fjac[4,4] = -fjac[4,1]
    end
    x = [3.; -1.; 0.; 1.]
    return name, f!, g!, x
end

function powell_badly_scaled()
    name = "powell_badly_scaled"
    const c1 = 1e4
    const c2 = 1.0001
    function f!(x, fvec)
        fvec[1] = c1*x[1]*x[2] - 1
        fvec[2] = exp(-x[1]) + exp(-x[2]) - c2
    end
    function g!(x, fjac)
        fjac[1,1] = c1*x[2]
        fjac[1,2] = c1*x[1]
        fjac[2,1] = -exp(-x[1])
        fjac[2,2] = -exp(-x[2])
    end
    x = [0.; 1.]
    return name, f!, g!, x
end

function wood()
    name = "wood"
    const c3 = 2e2
    const c4 = 2.02e1
    const c5 = 1.98e1
    const c6 = 1.8e2

    function f!(x, fvec)
        temp1 = x[2] - x[1]^2
        temp2 = x[4] - x[3]^2
        fvec[1] = -c3*x[1]*temp1 - (1 - x[1])
        fvec[2] = c3*temp1 + c4*(x[2] - 1) + c5*(x[4] - 1)
        fvec[3] = -c6*x[3]*temp2 - (1 - x[3])
        fvec[4] = c6*temp2 + c4*(x[4] - 1) + c5*(x[2] - 1)
    end

    function g!(x, fjac)
        fill!(fjac, 0)
        temp1 = x[2] - 3x[1]^2
        temp2 = x[4] - 3x[3]^2
        fjac[1,1] = -c3*temp1 + 1
        fjac[1,2] = -c3*x[1]
        fjac[2,1] = -2*c3*x[1]
        fjac[2,2] = c3 + c4
        fjac[2,4] = c5
        fjac[3,3] = -c6*temp2 + 1
        fjac[3,4] = -c6*x[3]
        fjac[4,2] = c5
        fjac[4,3] = -2*c6*x[3]
        fjac[4,4] = c6 + c4
    end
    x = [-3.; -1; -3; -1]
    return name, f!, g!, x
end

function helical_valley()
    name = "helical_valley"
    const tpi = 8*atan(1)
    const c7 = 2.5e-1
    const c8 = 5e-1

    function f!(x, fvec)
        if x[1] > 0
            temp1 = atan(x[2]/x[1])/tpi
        elseif x[1] < 0
            temp1 = atan(x[2]/x[1])/tpi + c8
        else
            temp1 = c7*sign(x[2])
        end
        temp2 = sqrt(x[1]^2+x[2]^2)
        fvec[1] = 10(x[3] - 10*temp1)
        fvec[2] = 10(temp2 - 1)
        fvec[3] = x[3]
    end

    function g!(x, fjac)
        temp = x[1]^2 + x[2]^2
        temp1 = tpi*temp
        temp2 = sqrt(temp)
        fjac[1,1] = 100*x[2]/temp1
        fjac[1,2] = -100*x[1]/temp1
        fjac[1,3] = 10
        fjac[2,1] = 10*x[1]/temp2
        fjac[2,2] = 10*x[2]/temp2
        fjac[2,3] = 0
        fjac[3,1] = 0
        fjac[3,2] = 0
        fjac[3,3] = 1
    end
    x = [-1.; 0; 0]
    return name, f!, g!, x
end

function watson(n::Integer)
    name = "watson"
    const c9 = 2.9e1

    function f!(x, fvec)
        fill!(fvec, 0)
        for i = 1:29
            ti = i/c9
            sum1 = 0.0
            temp = 1.0
            for j = 2:n
                sum1 += (j-1)*temp*x[j]
                temp *= ti
            end
            sum2 = 0.0
            temp = 1.0
            for j = 1:n
                sum2 += temp*x[j]
                temp *= ti
            end
            temp1 = sum1 - sum2^2 - 1
            temp2 = 2*ti*sum2
            temp = 1/ti
            for k = 1:n
                fvec[k] += temp*(k-1-temp2)*temp1
                temp *= ti
            end
        end
        temp = x[2] - x[1]^2 - 1
        fvec[1] += x[1]*(1-2temp)
        fvec[2] += temp
    end

    function g!(x, fjac)
        fill!(fjac, 0)
        for i = 1:29
            ti = i/c9
            sum1 = 0.0
            temp = 1.0
            for j = 2:n
                sum1 += (j-1)*temp*x[j]
                temp *= ti
            end
            sum2 = 0.0
            temp = 1.0

            for j = 1:n
                sum2 += temp*x[j]
                temp *= ti
            end
            temp1 = 2(sum1 - sum2^2 - 1)
            temp2 = 2*sum2
            temp = ti^2
            tk = 1.0

            for k = 1:n
                tj = tk
                for j = k:n
                    fjac[k,j] += tj*((Float64(k-1)/ti - temp2)*(Float64(j-1)/ti - temp2) - temp1)
                    tj *= ti
                end
                tk *= temp
            end
        end
        fjac[1,1] += 6x[1]^2 - 2x[2] + 3
        fjac[1,2] -= 2x[1]
        fjac[2,2] += 1
        for k = 1:n
            for j = k:n
                fjac[j,k] = fjac[k,j]
            end
        end
    end
    x = zeros(n)
    return name, f!, g!, x
end

function chebyquad(n::Integer)
    name = "chebyquad"
    const tk = 1/n

    function f!(x, fvec)
        fill!(fvec, 0)
        for j = 1:n
            temp1 = 1.0
            temp2 = 2x[j]-1
            temp = 2temp2
            for i = 1:n
                fvec[i] += temp2
                ti = temp*temp2 - temp1
                temp1 = temp2
                temp2 = ti
            end
        end
        iev = -1.0
        for k = 1:n
            fvec[k] *= tk
            if iev > 0
                fvec[k] += 1/(k^2-1)
            end
            iev = -iev
        end
    end

    function g!(x, fjac)
        for j = 1:n
            temp1 = 1.
            temp2 = 2x[j] - 1
            temp = 2*temp2
            temp3 = 0.0
            temp4 = 2.0
            for k = 1:n
                fjac[k,j] = tk*temp4
                ti = 4*temp2 + temp*temp4 - temp3
                temp3 = temp4
                temp4 = ti
                ti = temp*temp2 - temp1
                temp1 = temp2
                temp2 = ti
            end
        end
    end
    x = collect(1:n)/(n+1)
    return name, f!, g!, x
end

function brown_almost_linear(n::Integer)
    name = "brown_almost_linear"
    function f!(x, fvec)
        sum1 = sum(x) - (n+1)
        for k = 1:(n-1)
            fvec[k] = x[k] + sum1
        end
        fvec[n] = prod(x) - 1
    end

    function g!(x, fjac)
        fill!(fjac, 1)
        fjac[diagind(fjac)] = 2
        prd = prod(x)
        for j = 1:n
            if x[j] == 0.0
                fjac[n,j] = 1.
                for k = 1:n
                    if k != j
                        fjac[n,j] *= x[k]
                    end
                end
            else
                fjac[n,j] = prd/x[j]
            end
        end
    end
    x = 0.5*ones(n)
    return name, f!, g!, x
end

function discrete_boundary_value(n::Integer)
    name = "discrete_boundary_value"
    const h = 1/(n+1)

    function f!(x, fvec)
        for k = 1:n
            temp = (x[k] + k*h + 1)^3
            if k != 1
                temp1 = x[k-1]
            else
                temp1 = 0.0
            end
            if k != n
                temp2 = x[k+1]
            else
                temp2 = 0.0
            end
            fvec[k] = 2x[k] - temp1 - temp2 + temp*h^2/2
        end
    end

    function g!(x, fjac)
        for k = 1:n
            temp = 3*(x[k]+k*h+1)^2
            for j = 1:n
                fjac[k,j] = 0
            end
            fjac[k,k] = 2 + temp*h^2/2
            if k != 1
                fjac[k,k-1] = -1
            end
            if k != n
                fjac[k,k+1] = -1
            end
        end
    end
    x = collect(1:n)*h
    x = x .* (x .- 1)
    return name, f!, g!, x
end

function discrete_integral_equation(n::Integer)
    name = "discrete_integral_equation"
    const h = 1/(n+1)

    function f!(x, fvec)
        for k = 1:n
            tk = k*h
            sum1 = 0.0
            for j = 1:k
                tj = j*h
                sum1 += tj*(x[j] + tj + 1)^3
            end
            sum2 = 0.0
            kp1 = k+1
            if n >= kp1
                for j = kp1:n
                    tj = j*h
                    sum2 += (1-tj)*(x[j] + tj + 1)^3
                end
            end
            fvec[k] = x[k] + h*((1-tk)*sum1 + tk*sum2)/2
        end
    end

    function g!(x, fjac)
        for k = 1:n
            tk = k*h
            for j = 1:n
                tj = j*h
                fjac[k,j] = h*min(tj*(1-tk), tk*(1-tj))*3(x[j] + tj + 1)^2/2
            end
            fjac[k,k] += 1
        end
    end

    x = collect(1:n)*h
    x = x .* (x .- 1)

    return name, f!, g!, x
end

function trigonometric(n::Integer)
    name = "trigonometric"
    function f!(x, fvec)
        for j = 1:n
            fvec[j] = cos(x[j])
        end
        sum1 = sum(fvec)
        for k = 1:n
            fvec[k] = n+k-sin(x[k]) - sum1 - k*fvec[k]
        end
    end

    function g!(x, fjac)
        for j = 1:n
            temp = sin(x[j])
            for k = 1:n
                fjac[k,j] = temp
            end
            fjac[j,j] = (j+1)*temp - cos(x[j])
        end
    end
    x = ones(n)/n
    return name, f!, g!, x
end

function variably_dimensioned(n::Integer)
    name = "variably_dimensioned"
    function f!(x, fvec)
        sum1 = 0.0
        for j = 1:n
            sum1 += j*(x[j]-1)
        end
        temp = sum1*(1+2sum1^2)
        for k = 1:n
            fvec[k] = x[k] - 1 + k*temp
        end
    end

    function g!(x, fjac)
        sum1 = 0.0
        for j = 1:n
            sum1 += j*(x[j]-1)
        end
        temp = 1 + 6sum1^2
        for k = 1:n
            for j = k:n
                fjac[k,j] = k*j*temp
                fjac[j,k] = fjac[k,j]
            end
            fjac[k,k] += 1
        end
    end
    x = collect(1:n)/n
    return name, f!, g!, x
end

function broyden_tridiagonal(n::Integer)
    name = "broyden_tridiagonal"
    function f!(x, fvec)
        for k = 1:n
            temp = (3-2x[k])*x[k]
            if k != 1
                temp1 = x[k-1]
            else
                temp1 = 0.0
            end
            if k != n
                temp2 = x[k+1]
            else
                temp2 = 0.0
            end
            fvec[k] = temp - temp1 - 2temp2 + 1
        end
    end

    function g!(x, fjac)
        fill!(fjac, 0)
        for k = 1:n
            fjac[k,k] = 3-4x[k]
            if k != 1
                fjac[k,k-1] = -1
            end
            if k != n
                fjac[k,k+1] = -2
            end
        end
    end
    x = -ones(n)
    return name, f!, g!, x
end

function broyden_banded(n::Integer)
    name = "broyden_banded"
    const ml = 5
    const mu = 1

    function f!(x, fvec)
        for k = 1:n
            k1 = max(1, k-ml)
            k2 = min(k+mu, n)
            temp = 0.0
            for j = k1:k2
                if j != k
                    temp += x[j]*(1+x[j])
                end
            end
            fvec[k] = x[k]*(2+5x[k]^2) + 1 - temp
        end
    end

    function g!(x, fjac)
        fill!(fjac, 0)
        for k = 1:n
            k1 = max(1, k-ml)
            k2 = min(k+mu, n)
            for j = k1:k2
                if j != k
                    fjac[k,j] = -(1+2x[j])
                end
            end
            fjac[k,k] = 2+15x[k]^2
        end
    end
    x = -ones(n)
    return name, f!, g!, x
end



for matrix in (:dense, :sparse)
    for (solver, solver_abbr) in ((:qr, :qr), (:iterative, :iter))
        if (matrix == :sparse && solver == :qr)
            # tests below
            continue
        else
            for (method, method_abbr) in ((:dogleg, :dl), (:levenberg_marquardt, :lm))
                alltests = [rosenbrock(); 
                powell_singular(); powell_badly_scaled(); 
                wood();
                helical_valley(); 
                watson(6); watson(9);
                chebyquad(5); chebyquad(6); chebyquad(7); chebyquad(9);
                brown_almost_linear(10); brown_almost_linear(30); brown_almost_linear(40);
                discrete_boundary_value(10);
                discrete_integral_equation(1); discrete_integral_equation(10);
                trigonometric(10); variably_dimensioned(10); 
                broyden_tridiagonal(10); broyden_banded(10)]
                for (name, f!, g!, x) in alltests
                    fcur = similar(x)
                    J = Array(Float64, length(x), length(x))
                    if matrix == :sparse
                        J = sparse(J)
                    end
                    nls = LeastSquaresProblem(x, fcur, f!, J, g!)
                    r = optimize!(nls, method = method, solver = solver)
                    @printf("%-6s %4s %2s %30s %5d %5d   %5d   %10e\n", matrix, solver_abbr, method_abbr, name, r.iterations, r.f_calls, r.g_calls, r.ssr)
                    @test r.converged
                    @test r.ssr <= 1e-3
                end
            end
        end
    end
end

# test sparse factorization
alltests =  [rosenbrock(); 
powell_singular(); powell_badly_scaled(); 
wood();
helical_valley(); 
watson(6); watson(9);
chebyquad(5); chebyquad(6); chebyquad(7); chebyquad(9);
brown_almost_linear(10); brown_almost_linear(30); brown_almost_linear(40);
discrete_boundary_value(10);
discrete_integral_equation(1); discrete_integral_equation(10);
trigonometric(10); variably_dimensioned(10); 
broyden_tridiagonal(10); broyden_banded(10)]

for (name, f!, g!, x) in alltests
    fcur = similar(x)
    J = ones(length(x), length(x))
    J = sparse(J)
    fill!(J.nzval, 0)     
    g!(x, J)
    nls = LeastSquaresProblem(x, fcur, f!, J, g!)
    # try because g! may change symbolic factorizations (see Julia issue #9906)
    try
        r = optimize!(nls, method = :dogleg, solver = :cholesky)
        @test r.converged
        @test r.ssr <= 1e-3
        @printf("%-6s %4s %2s %30s %5d %5d   %5d   %10e\n", :sparse, :fact, :dl, name, r.iterations, r.f_calls, r.g_calls, r.ssr)
    catch y
        @test isa(y, ErrorException)
    end
end


# test dense cholesky
for (method, method_abbr) in ((:levenberg_marquardt, :lm), (:dogleg, :dl))
    alltests = [rosenbrock(); 
    powell_singular(); powell_badly_scaled(); 
    wood();
    helical_valley(); 
    watson(6); 
    chebyquad(5); chebyquad(6); chebyquad(7); chebyquad(9);
    brown_almost_linear(10);# brown_almost_linear(30);
    discrete_boundary_value(10);
    discrete_integral_equation(1); discrete_integral_equation(10);
    trigonometric(10); variably_dimensioned(10); 
    broyden_tridiagonal(10); broyden_banded(10)]
    for (name, f!, g!, x) in alltests
        fcur = similar(x)
        J = Array(Float64, length(x), length(x))
        nls = LeastSquaresProblem(x, fcur, f!, J, g!)
        r = optimize!(nls, method = method, solver = :cholesky)
        @printf("%-6s %4s %2s %30s %5d %5d   %5d   %10e\n", :dense, :chol, method_abbr, name, r.iterations, r.f_calls, r.g_calls, r.ssr)
        @test r.converged
        @test r.ssr <= 1e-3
    end
end


#test autodiff
for (method, method_abbr) in ((:dogleg, :dl), (:levenberg_marquardt, :lm))
    alltests = [rosenbrock(); 
    powell_singular(); powell_badly_scaled(); 
    wood();
    helical_valley(); 
    watson(6); watson(9);
    chebyquad(5); chebyquad(6); chebyquad(7); chebyquad(9);
    brown_almost_linear(10); brown_almost_linear(30); brown_almost_linear(40);
    discrete_boundary_value(10);
    discrete_integral_equation(1); discrete_integral_equation(10);
    trigonometric(10); variably_dimensioned(10); 
    broyden_tridiagonal(10); broyden_banded(10)]
    for (name, f!, g!, x) in alltests
        fcur = similar(x)
        J = Array(Float64, length(x), length(x))
        nls = LeastSquaresProblem(x, fcur, f!, J)
        r = optimize!(nls, method = method)
        @printf("%-10s %4s %2s %30s %5d %5d   %5d   %10e\n", :autodiff, :fact, method_abbr, name, r.iterations, r.f_calls, r.g_calls, r.ssr)
        @test r.converged
        @test r.ssr <= 1e-3
    end
end



# check default are correctly 
name, f!, g!, x = wood()
fcur = similar(x)
J = ones(length(x), length(x))
nls = LeastSquaresProblem(x, fcur, f!, J, g!)
result = optimize!(nls)
@test result.method == "dogleg"
nls = LeastSquaresProblem(x, fcur, f!, sparse(J), g!)
result = optimize!(nls)
@test result.method == "levenberg_marquardt"

