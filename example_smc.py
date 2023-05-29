# %% Use the particles package to achive SMC
# 2023-05-29
import particles
from particles import smc_samplers as ssp
from particles import distributions as dists
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from particles.smc_samplers import TemperingBridge, AdaptiveTempering


class ToyModel(ssp.StaticModel):
    def logpyt(self, theta, t):  # density of Y_t given theta and Y_{0:t-1}
        print(self.data.shape)
        print(theta["mu"].shape)
        return stats.norm.logpdf(
            self.data[t] - theta["mu"], loc=0, scale=theta["sigma"]
        )

        # return stats.norm.logpdf(residual(my_data[t], theta), loc=0., scale=0.)


T = 300
my_data = stats.norm.rvs(loc=10, scale=10, size=T)  # simulated data
# my_prior = dists.StructDist({"mu": dists.Normal(scale=10.0), "sigma": dists.Gamma(), "sigma2": dists.Normal()})
my_prior = dists.StructDist({"mu": dists.Normal(scale=100.0), "sigma": dists.Gamma()})

my_static_model = ToyModel(data=my_data, prior=my_prior)
my_ibis = ssp.IBIS(my_static_model, len_chain=50)
my_alg = particles.SMC(fk=my_ibis, N=100, store_history=True, verbose=True)
my_alg.run()
plt.style.use("ggplot")
# for i, p in enumerate(['mu', 'sigma', 'sigma2']):
for i, p in enumerate(["mu", "sigma"]):
    plt.subplot(1, 3, i + 1)
    for t in [1, int(np.floor(T / 2)), T - 1]:
        plt.hist(
            my_alg.hist.X[t].theta[p],
            weights=my_alg.hist.wgts[t].W,
            label="t=%i" % t,
            alpha=0.5,
            density=True,
        )
    plt.xlabel(p)
plt.legend()


# %%

data = np.random.normal(0, 1, [400,2])


class ToyModel(ssp.StaticModel):
    def logpyt(self, theta, t):
        return stats.norm.logpdf(
            (self.data[t][0] - theta["lambd"]) * (self.data[t][1] - theta["sigma"]),
            loc=0.0,
            scale=1.0,
        )


my_prior = dists.StructDist(
    {
        "lambd": dists.Normal(scale=1),
        # 'sigma': dists.Gamma()})
        "sigma": dists.Normal(scale=1),
    }
)
my_static_model = ToyModel(data=data, prior=my_prior)


fk_tempering = ssp.AdaptiveTempering(my_static_model)
my_temp_alg = particles.SMC(fk=fk_tempering, N=1000, ESSrmin=1.0, verbose=True)
my_temp_alg.run()
import seaborn as sb

for i, p in enumerate(["lambd", "sigma"]):
    plt.subplot(1, 2, i + 1)
    sb.histplot(my_temp_alg.X.theta[p], stat="density")
    plt.xlabel(p)
# %%
