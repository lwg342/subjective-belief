# %%
import numpy as np
from scipy.optimize import minimize
from wlpy.covariance import hac_cov
import pandas as pd
from utils import (
    MCMCBase,
    AdaptiveSampler,
    VectorMCMC,
    VectorMCMCKernel,
    BaseLogLike,
    uniform,
)


def residual_function(params, data):
    lambd, gamma = params
    consumption_growth = data[0]
    ret = data[1:]
    ret_excess_moment = 0.95 * (consumption_growth ** (-gamma)) * (ret)
    # rf_residual = 0.95 * (consumption_growth ** (-gamma)) * (rf) - 1.0
    residual_derivative = ret_excess_moment * (-gamma) / consumption_growth
    relative_entropy_weight = np.exp(lambd * ret_excess_moment)
    # relative_entropy_weight = 1.0
    return np.concatenate(
        [
            relative_entropy_weight * ret_excess_moment,
            relative_entropy_weight * lambd * residual_derivative,
        ]
    )


def objective_function(params, data, weighting_matrix=None):
    residual_mean = np.mean(residual_function(params, data), axis=1)
    if weighting_matrix is None:
        weighting_matrix = np.eye(residual_mean.shape[0])
    return np.dot(np.dot(residual_mean, weighting_matrix), residual_mean)


# %% Prepare data
consumption_df = pd.read_csv(
    "without_garbage_full.csv", delimiter="\t", index_col="year"
)
ff3_df = pd.read_csv("FF3_annual.CSV", index_col="year")


def extract_data(start_year, consumption_measure, return_measure):
    rf = (
        ff3_df.loc[(ff3_df.index >= start_year) & (ff3_df.index <= 2014)][
            "RF"
        ].to_numpy()
        / 100.0
        + 1.0
    )
    consumption_growth = np.array(
        consumption_df.loc[
            (consumption_df.index >= start_year) & (consumption_df.index <= 2014)
        ][consumption_measure]
        + 1.0
    )

    ret = np.array(
        consumption_df.loc[
            (consumption_df.index >= start_year) & (consumption_df.index <= 2014)
        ][return_measure]
    )
    ret2 = np.array(
        ff3_df.loc[(ff3_df.index >= start_year) & (ff3_df.index <= 2014)]["SMB"]
    )
    ret3 = np.array(
        ff3_df.loc[(ff3_df.index >= start_year) & (ff3_df.index <= 2014)]["HML"]
    )
    data = np.array([consumption_growth, ret, ret2, ret3, rf])
    return data


# data = extract_data(1960, "UNFIL-N&S", "MKT_DECfs")  # Full sample 1928, post war 1960
# data = extract_data(1960, "UNFIL-N&S", "MKT_Tafs")  # Full sample 1928, post war 1960
data = extract_data(1960, "NIPA-N&S", "MKT_Tafs")  # Full sample 1928, post war 1960

# %%
params_guess = np.array([0.0, 20.0])
result = minimize(
    objective_function,
    params_guess,
    args=(data,),
    bounds=[(-1, 1), (0, 200)],
    method="L-BFGS-B",
)
print(f"First Estimates: {result.x}")
weighting_matrix = np.linalg.inv(hac_cov(residual_function(result.x, data), lags=5))
result = minimize(
    objective_function,
    result.x,
    args=(data, weighting_matrix),
    bounds=[(-1, 1), (0, 200)],
    method="L-BFGS-B",
)
print(f"Second Estimates: {result.x}")

# %%
import matplotlib.pyplot as plt

x = np.linspace(0, 100, 1000)
y = [objective_function(np.array([0.0, i]), data) for i in x]
y2 = [objective_function(np.array([-0.001, i]), data) for i in x]
plt.plot(x, y)
plt.plot(x, y2)


# %%
def probability_divergence(p, eta=0):
    if eta == 0:
        return np.mean(p * np.log(p))
    else:
        raise NotImplementedError("Not implemented yet")


M = np.exp(result.x[0] * residual_function(result.x, data)[0]) / np.mean(
    np.exp(result.x[0] * residual_function(result.x, data)[0])
)
print(f"Probability divergence: {probability_divergence(M)}")


# %%
from particles.smc_samplers import TemperingBridge, AdaptiveTempering
import particles
from particles import smc_samplers as ssp
from particles import distributions as dists
from scipy import stats
import seaborn as sns
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
