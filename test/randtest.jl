module RandTest

using GLMCopula, Random, Statistics, Test

@testset "Normal(0,1) * (1 + 0.5 x^2)" begin
d = ContinuousUnivariateCopula(Normal(), 1.0, 0.0, 0.5)
@test d.c ≈ 2/3
@test mean(d) == 0
@test var(d) ≈ 5/3
@test minimum(d) == -Inf
@test maximum(d) == Inf
@test cdf(d, -Inf) == 0
@test cdf(d, 0) == 0.5
@test cdf(d, Inf) == 1

Random.seed!(123)
nsample = 1_000_000
@info "sample $nsample points"
s = Vector{Float64}(undef, nsample)
rand!(d, s) # compile
@time rand!(d, s)
println("sample mean = $(mean(s)); theoretical mean = $(mean(d))")
println("sample var = $(var(s)); theoretical var = $(var(d))")
end

end
