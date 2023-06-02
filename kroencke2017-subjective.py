# %%
import numpy as np
from scipy.optimize import minimize
from wlpy.covariance import hac_cov
import pandas as pd


def expand_to_2d(arr):
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    else:
        return arr


def residual_function(params, data):
    ll_mkt, ll_smb, ll_hml, ll_Rf, gamma = params


    consumption_growth, ret_mkt, ret_smb, ret_hml, ret_Rf = data[:5]
    
    # ll = np.array([[ll_mkt, ll_smb, ll_hml, ll_Rf]])
    # ll = np.array([[ll_mkt, ll_smb, ll_hml]])
    # ret = np.stack([ret_mkt, ret_smb, ret_hml])
    ll = np.array([[ll_mkt, ll_Rf]])
    # ll = np.array([[0.0]])
    # ll = np.array([[0,0,0,0]])
    ret = np.stack([ret_mkt])

    ret_moment = expand_to_2d(0.95 * (consumption_growth ** (-gamma)) * (ret))
    rf_moment = expand_to_2d( 0.95 * (consumption_growth ** (-gamma)) * (ret_Rf) - 1.0)

    all_moment = np.concatenate([ret_moment, rf_moment])
    # all_moment = np.concatenate([ret_moment])

    residual_derivative = -1.0 * all_moment * np.log(consumption_growth)

    # relative_entropy_weight = np.exp(np.clip(ll @ all_moment,-300, 500))
    relative_entropy_weight = np.exp(ll @ all_moment)
    # relative_entropy_weight = 1.0
    # print(np.max(all_moment))
    return np.concatenate(
        [
            relative_entropy_weight * all_moment,
            relative_entropy_weight * (ll @ residual_derivative),
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
    ret2 = (
        np.array(
            ff3_df.loc[(ff3_df.index >= start_year) & (ff3_df.index <= 2014)]["SMB"]
        )
        / 100.0
    )
    ret3 = (
        np.array(
            ff3_df.loc[(ff3_df.index >= start_year) & (ff3_df.index <= 2014)]["HML"]
        )
        / 100.0
    )
    data = np.array([consumption_growth, ret, ret2, ret3, rf])
    return data


# Full sample 1928, post war 1960
data = extract_data(1960, "UNFIL-N&S", "MKT_DECfs")
data = extract_data(1928, "UNFIL-N&S", "MKT_DECfs")
# data = extract_data(1928, "NIPA-N&S", "MKT_DECfs")
# data = extract_data(1928, "UNFIL-N&S", "MKT_Tafs")
# data = extract_data(1960, "Q4-N&S", "MKT_DECfs")
# data = extract_data(1960, "PJ-N&S", "MKT_DECfs")
# data = extract_data(1960, "UNFIL-N&S", "MKT_DECfs")
# data = extract_data(1960, "UNFIL-N&S", "MKT_Tafs")
# data = extract_data(1960, "NIPA-N&S", "MKT_DECfs")

# %%
params_guess = np.array([0.0, 0.0, 0.0, 0.0, 15.0])
result = minimize(
    objective_function,
    params_guess,
    args=(data,),
    # bounds=[(-10, 10), (0, 200)],
    method="L-BFGS-B",
)
print(f"First Estimates: {result.x}")
weighting_matrix = np.linalg.inv(hac_cov(residual_function(result.x, data), lags=5))
result = minimize(
    objective_function,
    result.x,
    args=(data, weighting_matrix),
    # bounds=[(-1, 1), (0, 200)],
    method="L-BFGS-B",
)
print(f"Second Estimates: {result.x}")
print(f"Objective function: {result.fun}")
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


def objective_fixed_data(params, weighting_matrix=None):
    residual_mean = np.mean(residual_function(params, data), axis=1)
    if weighting_matrix is None:
        weighting_matrix = np.eye(residual_mean.shape[0])
    return np.dot(np.dot(residual_mean, weighting_matrix), residual_mean)


def vectorise_objective_fixed_data(params, weighting_matrix=None):
    return np.apply_along_axis(objective_fixed_data, 1, params, weighting_matrix)


class ToyBridge(TemperingBridge):
    def logtarget(self, theta):
        rslt = vectorise_objective_fixed_data(theta)
        return -rslt


# %%
base_dist = dists.MvNormal(loc=[0.0, 0.0, 0.0, 0.0, 20.0], cov=np.eye(5))

toy_bridge = ToyBridge(base_dist=base_dist)
fk_tpr = AdaptiveTempering(model=toy_bridge, len_chain=50)
alg = particles.SMC(fk=fk_tpr, N=20)
alg.run()


fig, axes = plt.subplots(1, 2)


sns.histplot(alg.X.theta[:, 0], stat="density", kde=True, ax=axes[1])
axes[1].set_title("SMC Distribution of Lambda")
sns.histplot(alg.X.theta[:, -1], stat="density", kde=True, ax=axes[0])
axes[0].set_title("SMC Distribution of Gamma")

plt.tight_layout()
plt.show()
# %%
import numpy as np

# Create a NumPy array
arr = alg.X.theta[:, -1]

print("2.5th percentile:", np.percentile(arr, 2.5))
print("97.5th percentile:", np.percentile(arr, 97.5))


# %%
