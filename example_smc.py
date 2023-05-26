# %%
from utils import MCMCBase,AdaptiveSampler, VectorMCMC, VectorMCMCKernel,BaseLogLike, uniform 
import numpy as np


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


sample_size = 200
k = 2

true_param = np.array([[2, 5, 8]])
x = np.random.uniform(0, 3, sample_size)


def eval_model(theta):
    # time.sleep(0.1) # artificial slowdown to show off progress bar
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    c = theta[:, 2, None]
    return a + x*b + x**2*c 


# %%

noisy_data = eval_model(true_param) + np.random.normal(0, 2, sample_size)
priors = [uniform(0.0, 100.0), uniform(0.0, 100.0), uniform(0.0, 100.0), uniform(0.0, 100.0)]
model = VectorMCMC(
    eval_model,
    noisy_data,
    priors=priors,
    log_like_args=None,
    log_like_func=NewLogLikelihood,
)
mcmc_kernel = VectorMCMCKernel(model, param_order=("lambd1", "lambd2", "c", "mu"))
smc = AdaptiveSampler(mcmc_kernel)
step_list, mll_list = smc.sample(
    num_particles=200, num_mcmc_samples=1000, target_ess=0.1, progress_bar=True
)
step_list[-1].params

# plt.plot(step_list[-1].params[:, 0], step_list[-1].params[:, 1], "o")
# %% 
print(step_list[-1].params[:, 0].mean())
print(step_list[-1].params[:, 1].mean())
print(step_list[-1].params[:, 2].mean())
print(step_list[-1].params[:, 3].mean())
# %%
