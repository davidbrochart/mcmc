import sys
sys.path.append('../mcmc')
import smc
import dist
import numpy as np
import matplotlib.pyplot as plt

chains = 500

n = 4

mu1 = np.ones(n) * (1. / 2)
mu2 = -mu1

stdev = 0.1
sigma = np.power(stdev, 2) * np.eye(n)
isigma = np.linalg.inv(sigma)
dsigma = np.linalg.det(sigma)

w1 = 0.1
w2 = (1 - w1)

def two_gaussians(x):
    log_like1 = - 0.5 * n * np.log(2 * np.pi) \
                - 0.5 * np.log(dsigma) \
                - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
    log_like2 = - 0.5 * n * np.log(2 * np.pi) \
                - 0.5 * np.log(dsigma) \
                - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
    logp = np.log(w1 * np.exp(log_like1) + w2 * np.exp(log_like2))
    return logp

def prior_logp(v):
    logp = sum([dist.logp_from_pdf(pdf, v[i]) for i, pdf in enumerate(x)])
    return logp

x = [dist.uniform_pdf(-2, 2)] * n
posterior = smc.smc(x, two_gaussians, prior_logp, cores=4)

plt.figure()
for i in range(n):
    ax = plt.subplot(100*n+10+i+1)
    cdf = dist.cdf_from_samples(posterior[:, i])
    pdf = dist.pdf_from_cdf(cdf)
    plt.plot(*pdf)
plt.show()
