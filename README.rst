This is a simple implementation of Markov Chain Monte Carlo sampler using the Metropolis-Hastings algorithm.

```python
import mcmc
# just provide initial values (q0) and a function returning the log-probability (lnprob)
sampler = mcmc.sampler(q0, lnprob)
samples = sampler.sample(1000)
```

![Screenshot](examples/triangle.png)
