using Test, Turing, FlexiChains, StatsPlots
using FlexiChains: Extra, Prefixed, Parameter
using DynamicPPL, MCMCChains, AbstractMCMC, LinearAlgebra

# This test file just runs the Turing-related examples in 
# https://juliabayes.org/FlexiChains.jl/v0.6
# to make sure that they aren't broken on v0.7.

@testset "Turing + FlexiChains 0.7 compat" begin
    @testset "turing.md" begin
        # https://juliabayes.org/FlexiChains.jl/v0.6/turing
        y = [28, 8, -3, 7, -1, 1, 18, 12]
        sigma = [15, 10, 16, 11, 9, 11, 10, 18]
        @model function eight_schools(y, sigma)
            mu ~ Normal(0, 5)
            tau ~ truncated(Cauchy(0, 5); lower=0)
            theta ~ MvNormal(fill(mu, length(y)), tau^2 * I)
            for i in eachindex(y)
                y[i] ~ Normal(theta[i], sigma[i])
            end
            return (mu=mu, tau=tau)
        end
        model = eight_schools(y, sigma)
        chain = sample(model, NUTS(), 5)

        chain[@varname(mu)]
        chain[@varname(theta)]
        chain[@varname(theta), stack=true]
        chain[@varname(theta[1])]
        chain[@varname(mu), iter=2:4, chain=1]
        chain[@varname(mu), iter=Not(At(7)), chain=At(1)]

        chain[Extra(:logjoint)]
        chain[:mu] # parameter
        chain[:logjoint] # other key

        @model inner() = x ~ MvNormal(zeros(2), I)
        @model function outer()
            a ~ to_submodel(inner())
            return nothing
        end
        pfx_chain = sample(outer(), MH(), 5)
        pfx_chain[Prefixed(@varname(x))]
        pfx_chain[Prefixed(@varname(x[1]))]

        summarystats(chain)
        ss = summarystats(chain)
        ss[@varname(mu)]
        ss[@varname(mu), stat=:mean]
        DimArray(ss)
        summarystats(chain; split_varnames=false)
        mn = mean(chain)
        mn[@varname(mu)]
        mean(chain; dims=:iter)[@varname(mu)]
        chn1 = sample(model, NUTS(), 10; save_state=true)
        chn2 = sample(model, NUTS(), 10; initial_state=only(FlexiChains.last_sampler_state(chn1)))
        combined_chn = vcat(chn1, chn2)
        chn3 = sample(model, MH(), 5; initial_params=InitFromParams(chain, 5, 1))
        returned(model, chain)

        plot(chain, [@varname(tau), @varname(theta[1])])

        @test true
    end

    @testset "mcmcchains.md" begin
        # https://juliabayes.org/FlexiChains.jl/v0.6/mcmcchains
        @model function f()
            x ~ Normal()
            y ~ Bernoulli()
            z ~ MvNormal(zeros(2), I)
        end
        samples = [DynamicPPL.ParamsWithStats(rand(f()), (; logp=rand())) for _ in 1:100, _ in 1:3]
        mc = AbstractMCMC.from_samples(MCMCChains.Chains, samples)
        FlexiChains.from_mcmcchains(mc)

        fc = FlexiChains.from_mcmcchains(
            mc,
            (
                Parameter(@varname(x)),
                Parameter(@varname(y)),
                Parameter(@varname(z)) => (2,),
                Extra(:logp),
            ),
        )
        fc = FlexiChains.transform_values(fc, @varname(y) => Bool)
        MCMCChains.Chains(fc)

        @test true
    end
end
