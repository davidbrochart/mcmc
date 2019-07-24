from .dist import cdf_from_pdf, sample_from_cdf
import numpy as np
import scipy.linalg
from tqdm import tqdm
import multiprocessing as mp

def smc(prior_pdf, likelihood_logp, prior_logp, draws=5000, step=None, cores=1, dask_client=None):
    """
    Sequential Monte Carlo sampling

    Parameters
    ----------
    prior_pdf : list of 2-dimension arrays
        The prior PDFs.
    likelihood_logp : function
        The model log-likelihood function.
    prior_logp :  function
        The prior log-likelihood function.
    draws : int
        The number of samples to draw from the posterior (i.e. last stage). And
        also the number of independent Markov Chains. Defaults to 5000.
    step : :class:`SMC`
        SMC initialization object
    cores : int
        Number of CPU cores to use. Multiprocessing is used when cores > 1.
    dask_client : dask.distributed.Client
        Dask distribute client to use for running e.g. on a cluster.
    """

    if step is None:
        step = SMC()
    accepted = 0
    acc_rate = 1.0
    proposed = draws * step.n_steps
    stage = 0 
    beta= 0
    marginal_likelihood = 1
    prior_cdf = [cdf_from_pdf(pdf) for pdf in prior_pdf]
    posterior = np.array(list(zip(*[sample_from_cdf(cdf, draws) for cdf in prior_cdf])))
    logp = likelihood_logp(posterior[0])
    if (type(logp) is list) or (type(logp) is tuple):
        has_blobs = True
    else:
        has_blobs = False
    
    while beta < 1:
        # compute plausibility weights (measure fitness)
        if dask_client:
            dask_posterior = dask_client.scatter(list(posterior), broadcast=True)
            futures = dask_client.map(likelihood_logp, dask_posterior)
            results = dask_client.gather(futures)
        elif cores > 1:
            pool = mp.Pool(processes=cores)
            results = pool.starmap(
                likelihood_logp,
                [(sample,) for sample in posterior]
            )
        else:
            results = [likelihood_logp(sample) for sample in posterior]
        if has_blobs:
            results = [res[0] for res in results]
        likelihoods = np.array(results)
        beta, old_beta, weights, sj = _calc_beta(beta, likelihoods, step.threshold)
        marginal_likelihood *= sj
        # resample based on plausibility weights (selection)
        resampling_indexes = np.random.choice(np.arange(draws), size=draws, p=weights)
        posterior = posterior[resampling_indexes]
        likelihoods = likelihoods[resampling_indexes]
    
        # compute proposal distribution based on weights
        covariance = _calc_covariance(posterior, weights)
        proposal = MultivariateNormalProposal(covariance)
    
        # compute scaling (optional) and number of Markov chains steps (optional), based on the
        # acceptance rate of the previous stage
        if (step.tune_scaling or step.tune_steps) and stage > 0:
            _tune(acc_rate, proposed, step)
    
        print("Stage: {:d} Beta: {:.3f} Steps: {:d}".format(stage, beta, step.n_steps))
        # Apply Metropolis kernel (mutation)
        proposed = draws * step.n_steps
        priors = np.array([prior_logp(sample) for sample in posterior])
        tempered_logp = priors + likelihoods * beta
        deltas = proposal(step.n_steps) * step.scaling
    
        parameters = (
            proposal,
            step.scaling,
            accepted,
            step.n_steps,
            prior_logp,
            likelihood_logp,
            beta,
            has_blobs,
        )
        
        if dask_client:
            dask_posterior = dask_client.scatter(list(posterior), broadcast=True)
            dask_tempered_logp = dask_client.scatter(list(tempered_logp), broadcast=True)
            dask_parameters = dask_client.scatter(parameters, broadcast=True)
            futures = dask_client.map(_metrop_kernel, dask_posterior, dask_tempered_logp, *[[param] * draws for param in dask_parameters], pure=False)
            results = dask_client.gather(futures)
        elif cores > 1:
            pool = mp.Pool(processes=cores)
            results = pool.starmap(
                _metrop_kernel,
                [(posterior[draw], tempered_logp[draw], *parameters) for draw in range(draws)],
            )
        else:
            results = [
                _metrop_kernel(posterior[draw], tempered_logp[draw], *parameters)
                for draw in tqdm(range(draws))
            ]
    
        if beta == 1 and has_blobs:
            posterior, acc_list, blobs = zip(*results)
        else:
            posterior, acc_list = zip(*results)
        posterior = np.array(posterior)
        acc_rate = sum(acc_list) / proposed
        stage += 1

    if has_blobs:
        return posterior, blobs
    else:
        return posterior

def metrop_select(mr, q, q0):
    """Perform rejection/acceptance step for Metropolis class samplers.

    Returns the new sample q if a uniform random number is less than the
    metropolis acceptance rate (`mr`), and the old sample otherwise, along
    with a boolean indicating whether the sample was accepted.

    Parameters
    ----------
    mr : float, Metropolis acceptance rate
    q : proposed sample
    q0 : current sample

    Returns
    -------
    q or q0
    """
    # Compare acceptance ratio to uniform random number
    if np.isfinite(mr) and np.log(np.random.uniform()) < mr:
        return q, True
    else:
        return q0, False

def _metrop_kernel(
    q_old,
    old_tempered_logp,
    proposal,
    scaling,
    accepted,
    n_steps,
    prior_logp,
    likelihood_logp,
    beta,
    has_blobs,
):
    """
    Metropolis kernel
    """
    deltas = proposal(n_steps) * scaling
    new_blob = False
    for n_step in range(n_steps):
        delta = deltas[n_step]

        q_new = q_old + delta

        if has_blobs:
            l_logp, blob_new = likelihood_logp(q_new)
        else:
            l_logp = likelihood_logp(q_new)
        new_tempered_logp = prior_logp(q_new) + l_logp * beta

        q_old, accept = metrop_select(new_tempered_logp - old_tempered_logp, q_new, q_old)
        if accept:
            accepted += 1
            old_tempered_logp = new_tempered_logp
            if has_blobs:
                blob_old = blob_new
                new_blob = True

    if beta == 1 and has_blobs:
        # blobs are not kept from previous calls of likelihood_logp
        # instead we re-compute them if needed
        if not new_blob:
            _, blob_old = likelihood_logp(q_old)
        return q_old, accepted, blob_old
    else:
        return q_old, accepted

def _calc_beta(beta, likelihoods, threshold=0.5):
    """
    Calculate next inverse temperature (beta) and importance weights based on
    current beta and tempered likelihood.

    Parameters
    ----------
    beta : float
        tempering parameter of current stage
    likelihoods : numpy array
        likelihoods computed from the model
    threshold : float
        Determines the change of beta from stage to stage, i.e.indirectly the
        number of stages, the higher the value of threshold the higher the
        number of stage. Defaults to 0.5.  It should be between 0 and 1.

    Returns
    -------
    new_beta : float
        tempering parameter of the next stage
    old_beta : float
        tempering parameter of the current stage
    weights : numpy array
        Importance weights (floats)
    sj : float
        Partial marginal likelihood
    """
    low_beta = old_beta = beta
    up_beta = 2.0
    rN = int(len(likelihoods) * threshold)

    while up_beta - low_beta > 1e-6:
        new_beta = (low_beta + up_beta) / 2.0
        weights_un = np.exp((new_beta - old_beta) * (likelihoods - likelihoods.max()))
        weights = weights_un / np.sum(weights_un)
        ESS = int(1 / np.sum(weights ** 2))
        if ESS == rN:
            break
        elif ESS < rN:
            up_beta = new_beta
        else:
            low_beta = new_beta
    if new_beta >= 1:
        new_beta = 1
    sj = np.exp((new_beta - old_beta) * likelihoods)
    weights_un = np.exp((new_beta - old_beta) * (likelihoods - likelihoods.max()))
    weights = weights_un / np.sum(weights_un)
    return new_beta, old_beta, weights, np.mean(sj)

def _calc_covariance(posterior, weights):
    """
    Calculate trace covariance matrix based on importance weights.
    """
    cov = np.cov(posterior, aweights=weights.ravel(), bias=False, rowvar=0)
    if np.isnan(cov).any() or np.isinf(cov).any():
        raise ValueError('Sample covariances not valid! Likely "draws" is too small!')
    return np.atleast_2d(cov)

class Proposal:
    def __init__(self, s):
        self.s = s

class MultivariateNormalProposal(Proposal):
    def __init__(self, s):
        n, m = s.shape
        if n != m:
            raise ValueError("Covariance matrix is not symmetric.")
        self.n = n
        self.chol = scipy.linalg.cholesky(s, lower=True)

    def __call__(self, num_draws=None):
        if num_draws is not None:
            b = np.random.randn(self.n, num_draws)
            return np.dot(self.chol, b).T
        else:
            b = np.random.randn(self.n)
            return np.dot(self.chol, b)

def _tune(acc_rate, proposed, step):
    """
    Tune scaling and/or n_steps based on the acceptance rate.

    Parameters
    ----------
    acc_rate: float
        Acceptance rate of the previous stage
    proposed: int
        Total number of proposed steps (draws * n_steps)
    step: SMC step method
    """
    if step.tune_scaling:
        # a and b after Muto & Beck 2008.
        a = 1 / 9
        b = 8 / 9
        step.scaling = (a + b * acc_rate) ** 2
    if step.tune_steps:
        acc_rate = max(1.0 / proposed, acc_rate)
        step.n_steps = min(step.max_steps, 1 + int(np.log(step.p_acc_rate) / np.log(1 - acc_rate)))

class SMC:
    R"""
    Sequential Monte Carlo step

    Parameters ---------- n_steps : int The number of steps of a Markov Chain.
    If `tune_steps == True` `n_steps` will be used for the first stage and the
    number of steps of the other stages will be determined automatically based
    on the acceptance rate and `p_acc_rate`.  The number of steps will never be
    larger than `n_steps`.  scaling : float Factor applied to the proposal
    distribution i.e. the step size of the Markov Chain. Only works if
    `tune_scaling == False` otherwise is determined automatically.  p_acc_rate
    : float Used to compute `n_steps` when `tune_steps == True`. The higher the
    value of `p_acc_rate` the higher the number of steps computed
    automatically. Defaults to 0.99. It should be between 0 and 1.
    tune_scaling : bool Whether to compute the scaling automatically or not.
    Defaults to True tune_steps : bool Whether to compute the number of steps
    automatically or not. Defaults to True threshold : float Determines the
    change of beta from stage to stage, i.e.indirectly the number of stages,
    the higher the value of `threshold` the higher the number of stages.
    Defaults to 0.5.  It should be between 0 and 1.  parallel : bool Distribute
    computations across cores if the number of cores is larger than 1 (see
    pm.sample() for details). Defaults to True.  dask_client:
    dask.distributed.Client or None Distribute computations through a Dask
    distributed scheduler (locally or on a cluster).  model :
    :class:`pymc3.Model` Optional model for sampling step. Defaults to None
    (taken from context).

    Notes ----- SMC works by moving from successive stages. At each stage the
    inverse temperature \beta is increased a little bit (starting from 0 up to
    1). When \beta = 0 we have the prior distribution and when \beta =1 we have
    the posterior distribution. So in more general terms we are always
    computing samples from a tempered posterior that we can write as:

    p(\theta \mid y)_{\beta} = p(y \mid \theta)^{\beta} p(\theta)

    A summary of the algorithm is:

     1. Initialize \beta at zero and stage at zero.  2. Generate N samples
     S_{\beta} from the prior (because when \beta = 0 the tempered posterior is
     the prior).  3. Increase \beta in order to make the effective sample size
     equals some predefined value (we use N*t, where t is 0.5 by default).  4.
     Compute a set of N importance weights W. The weights are computed as the
     ratio of the likelihoods of a sample at stage i+1 and stage i.  5. Obtain
     S_{w} by re-sampling according to W.  6. Use W to compute the covariance
     for the proposal distribution.  7. For stages other than 0 use the
     acceptance rate from the previous stage to estimate the scaling of the
     proposal distribution and n_steps.  8. Run N Metropolis chains (each one
     of length n_steps), starting each one from a different sample in S_{w}.
     9. Repeat from step 3 until \beta \ge 1.  10. The final result is a
     collection of N samples from the posterior.


    References
    ----------
    .. [Minson2013] Minson, S. E. and Simons, M. and Beck, J. L., (2013),
        Bayesian inversion for finite fault earthquake source models I- Theory
        and algorithm.  Geophysical Journal International, 2013, 194(3),
        pp.1701-1726, `link
        <https://gji.oxfordjournals.org/content/194/3/1701.full>`__

    .. [Ching2007] Ching, J. and Chen, Y. (2007).
        Transitional Markov Chain Monte Carlo Method for Bayesian Model
        Updating, Model Class Selection, and Model Averaging. J. Eng. Mech.,
        10.1061/(ASCE)0733-9399(2007)133:7(816), 816-832. `link
        <http://ascelibrary.org/doi/abs/10.1061/%28ASCE%290733-9399
        %282007%29133:7%28816%29>`__
    """

    def __init__(
        self,
        n_steps=25,
        scaling=1.0,
        p_acc_rate=0.99,
        tune_scaling=True,
        tune_steps=True,
        threshold=0.5,
    ):

        self.n_steps = n_steps
        self.max_steps = n_steps
        self.scaling = scaling
        self.p_acc_rate = 1 - p_acc_rate
        self.tune_scaling = tune_scaling
        self.tune_steps = tune_steps
        self.threshold = threshold
