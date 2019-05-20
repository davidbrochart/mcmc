import sys
sys.path.append('../mcmc')
import smc
import dist
import numpy as np
import matplotlib.pyplot as plt
import time

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
    time.sleep(0.001) # artificially slow the model down
    return logp

x_pdf = [dist.uniform_pdf(-2, 2)] * n
def prior_logp(values):
    logp = sum([dist.logp_from_pdf(pdf, v) for pdf, v in zip(x_pdf, values)])
    return logp

t0 = time.time()
posterior = smc.smc(x_pdf, two_gaussians, prior_logp, cores=4)
t1 = time.time()
print(f'Took {t1-t0} seconds')

plt.figure()
for i in range(n):
    ax = plt.subplot(100*n+10+i+1)
    cdf = dist.cdf_from_samples(posterior[:, i])
    pdf = dist.pdf_from_cdf(cdf)
    plt.plot(*pdf)
plt.show()
