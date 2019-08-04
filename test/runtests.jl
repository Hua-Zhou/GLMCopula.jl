module PkgTest

using InteractiveUtils, LinearAlgebra, Profile, Random
using BenchmarkTools, GLMCopula

Random.seed!(123)

n = 100  # number of observations
ns = rand(10:30, n) # ni in each observation
p = 3   # number of mean parameters
m = 2   # number of variance components
gcs = Vector{GaussianCopulaVC{Float64}}(undef, n)
# true parameter values
βtruth = ones(p)
σ2truth = collect(1.:m)
σ02truth = 1.0
for i in 1:n
    ni = ns[i]
    # set up covariance matrix
    V1 = convert(Matrix, Symmetric([Float64(i * (ni - j + 1)) for i in 1:ni, j in 1:ni])) # a pd matrix
    V1 ./= norm(V1) / sqrt(ni) # scale to have Frobenius norm sqrt(n)
    prob = fill(1/ni, ni)
    V2 = ni .* (Diagonal(prob) - prob * transpose(prob))
    V2 ./= norm(V2) / sqrt(ni) # scale to have Frobenious norm sqrt(n)
    Ω = σ2truth[1] * V1 + σ2truth[2] * V2 + σ02truth * I
    Ωchol = cholesky(Symmetric(Ω))
    # simulate design matrix
    X = [ones(ni) randn(ni, p-1)]
    # generate responses
    y = X * βtruth + Ωchol.L * randn(ni)
    # add to data
    gcs[i] = GaussianCopulaVC(y, X, [V1, V2])
end

gcm = GaussianCopulaVCModel(gcs)

@info "Initial point:"
init_β!(gcm)
@show gcm.β
@show gcm.τ
standardize_res!(gcm)
update_quadform!(gcm, true)
fill!(gcm.σ2, 1)
update_σ2!(gcm)
@show gcm.σ2
# @btime update_σ2!(gcm) setup=(fill!(gcm.σ2, 1))

# @show loglikelihood!(gcm.data[1], gcm.β, gcm.τ[1], gcm.σ2, true, false)
# @code_warntype loglikelihood!(gcm.data[1], gcm.β, gcm.τ[1], gcm.σ2, true, false)
# @btime loglikelihood!(gcm.data[1], gcm.β, gcm.τ[1], gcm.σ2, true, false)

@show loglikelihood!(gcm, true, false)
@show gcm.∇
# @code_warntype loglikelihood!(gcm, false, false)
# @code_llvm loglikelihood!(gcm, false, false)
# @btime loglikelihood!(gcm, true, false)

# Solvers:
# Ipopt.IpoptSolver(print_level=0)
# NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_CCSAQ, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_TNEWTON_PRECOND_RESTART, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_TNEWTON_PRECOND, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_TNEWTON_RESTART, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_TNEWTON, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_VAR1, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_VAR2, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LN_COBYLA, maxeval=10000)
# NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000)

@info "MLE:"
solver = Ipopt.IpoptSolver(print_level=0)
fit!(gcm, solver) # force compilation
@show gcm.β
@show gcm.τ
@show gcm.σ2
@show gcm.∇
@show loglikelihood!(gcm)
# @btime fit!(gcm, solver)

# Profile.clear()
# @profile begin
# for solver in [
#     # NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000),
#     NLopt.NLoptSolver(algorithm=:LD_CCSAQ, maxeval=4000),
#     NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000),
#     NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000),
#     # NLopt.NLoptSolver(algorithm=:LD_TNEWTON_PRECOND_RESTART, maxeval=4000),
#     #NLopt.NLoptSolver(algorithm=:LD_TNEWTON_PRECOND, maxeval=4000),
#     #NLopt.NLoptSolver(algorithm=:LD_TNEWTON_RESTART, maxeval=4000),
#     #NLopt.NLoptSolver(algorithm=:LD_TNEWTON, maxeval=4000),
#     # NLopt.NLoptSolver(algorithm=:LD_VAR1, maxeval=4000),
#     # NLopt.NLoptSolver(algorithm=:LD_VAR2, maxeval=4000),
#     Ipopt.IpoptSolver(print_level=3)
#     ]
#     @show solver
#     # fill!(vcm.σ2, 0.25) # re-set starting point
#     fit_mom!(vcm)
#     @time fit!(vcm, solver)
#     @show vcm.σ2
#     @show composite_loglikelihood!(vcm)
#     println()
# end
# end
# Profile.print(format=:flat)

end