import numpy as np
import sys
from scipy.stats import uniform

smc_path = "SMCPy/"
sys.path.append(smc_path)
from smcpy import AdaptiveSampler, VectorMCMC, VectorMCMCKernel
from smcpy.mcmc.mcmc_base import MCMCBase
from smcpy.log_likelihoods import BaseLogLike


def probability_divergence(p, eta=0):
    if eta == 0:
        return np.mean(p * np.log(p))
    else:
        raise NotImplementedError("Not implemented yet")
