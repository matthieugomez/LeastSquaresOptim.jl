
function ls_optim(x, fcur, f!, J, g!; 
                method = :dogleg,
                xtol::Number = 1e-32, ftol::Number = 1e-32, grtol::Number = 1e-8,
                iterations::Integer = 100, store_trace::Bool = false)
    xx = similar(x)
    copy!(xx, x)
    lsoptim!(xx, fcur, f!, J, g!;
    method = method, xtol = xtol, ftol = ftol, grtol = grtol,
    iterations = iterations, store_trace = store_trace)
end

function ls_optim!(x, fcur, f!, J, g!; 
                method = :dogleg,
                xtol::Number = 1e-32, ftol::Number = 1e-32, grtol::Number = 1e-8,
                iterations::Integer = 100, store_trace::Bool = false)
    ls_optim!(Val{method}, x, fcur, f!, J, g!;
        xtol = xtol, ftol = ftol, grtol = grtol, 
        iterations = iterations, store_trace = store_trace)
end

type LSResults
    name::AbstractString
    x::Any
    ssr::Real
    iter::Real
    converged::Bool
    x_converged::Bool
    xtol::Real
    f_converged::Bool
    ftol::Real
    gr_converged::Bool
    grtol::Real
    f_calls::Int
    g_calls::Int
    mul_calls::Int
end



function assess_convergence(δx,
                            ssr,
                            oldssr,
                            f,
                            xtmp,
                            xtol::Real,
                            ftol::Real,
                            grtol::Real)
    x_converged, f_converged, gr_converged = false, false, false

    if maxabs(δx) < xtol
        x_converged = true
    end
    # Absolute Tolerance
    # if abs(ssr - oldssr) < ftol
    # Relative Tolerance
    if abs(ssr - oldssr) / (abs(ssr) + ftol) < ftol || nextfloat(ssr) >= oldssr
        f_converged = true
    end

    if maxabs(xtmp) < grtol
        gr_converged = true
    end

    converged = x_converged || f_converged || gr_converged

    return x_converged, f_converged, gr_converged, converged
end

for name in (:Ac_mul_B!, :A_mul_B!)
    _name = parse("_$name")
    @eval begin
        $_name(y::Vector, X::Matrix, x::Vector) = $name(y, X, x)
        $_name(y, X, x) = $name(1.0, X, x, 0.0, y)
    end
end


function sumabs21!(v::Vector, A::Base.SparseMatrix.SparseMatrixCSC)
    for i in 1:length(v)
        v[i] = sumabs2(sub(nonzeros(A), nzrange(A, i)))
    end
end