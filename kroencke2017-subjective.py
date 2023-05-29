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
class NewLogLikelihood(BaseLogLike):
    def __init__(self, model, data, args):
        """
        Likelihood function for data with additive, iid errors sampled from a
        normal distribution with mean = 0 and std_dev = args. If args is None,
        assumes that the last column of inputs contains the std_dev value.
        """
        super().__init__(model, data, args)

    def __call__(self, inputs):
        std_dev = self._args
        if std_dev is None:
            std_dev = inputs[:, -1]
            inputs = inputs[:, :-1]
        var = std_dev**2

        output = self._get_output(inputs)

        return self._calc_normal_log_like(output, self._data, var)

    @staticmethod
    def _calc_normal_log_like(output, data, var):
        ssqe = np.sum((output - data) ** 2, axis=1)

        term1 = -np.log(2 * np.pi * var) * (output.shape[1] / 2.0)
        term2 = -1 / 2.0 * ssqe / var

        return term1 + term2


def eval_model(theta):
    # time.sleep(0.1) # artificial slowdown to show off progress bar
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    c = theta[:, 2, None]
    return a + x * b + x**2 * c
