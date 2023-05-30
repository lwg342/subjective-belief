# %% Use the particles package to achive SMC
# 2023-05-29
import matplotlib.pyplot as plt
import numpy as np
from particles.smc_samplers import TemperingBridge, AdaptiveTempering
import particles
from particles import smc_samplers as ssp
from particles import distributions as dists
from scipy import stats
import seaborn as sns


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

data = np.random.normal(0, 1, [400, 2])


class ToyModel(ssp.StaticModel):
    def logpyt(self, theta, t):
        print(theta.shape)
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

for i, p in enumerate(["lambd", "sigma"]):
    plt.subplot(1, 2, i + 1)
    sns.histplot(my_temp_alg.X.theta[p], stat="density")
    plt.xlabel(p)


# %%
class TestModel(ssp.TemperingModel):
    def logtarget(self, theta):
        pass


params = dists.MvNormal(scale=10.0, cov=np.eye(2)).rvs(1000)
data = data.T
print(data.shape)
print(params.shape)


# %% With Data input
def residual(params):
    rslt = (data[0] - params[0]) ** 2 + (data[1] - (params[1])) ** 2
    rslt = -0.5 * rslt
    return np.mean(rslt)


class ToyBridge(TemperingBridge):
    def logtarget(self, theta):
        # print(theta.shape)
        # rslt = -0.5 * np.sum(theta**2, axis=1)
        # rslt = -0.5 * (theta[:,0]**2 + theta[:,1]**2)
        rslt = np.apply_along_axis(residual, 1, theta)
        print(rslt.shape)
        # print((rslt))
        return rslt

data = np.random.normal(20, 1, [2,100])
base_dist = dists.MvNormal(scale=0.001, cov=np.eye(2))
toy_bridge = ToyBridge(base_dist=base_dist)
fk_tpr = AdaptiveTempering(model=toy_bridge, len_chain=100)
alg = particles.SMC(fk=fk_tpr, N=100)
alg.run()
sns.histplot(alg.X.theta[:, 1], stat="density")
sns.histplot(alg.X.theta[:, 0], stat="density")
# %%
