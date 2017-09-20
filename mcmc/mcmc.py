import numpy as np
try:
    from tqdm import tqdm
except:
    tqdm = None

class walker:
    def __init__(self, scale=1, tune_interval=100):
        self.scale = scale
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0
    def step(self, q0):
        if self.steps_until_tune == 0:
            acc_rate = self.accepted / self.tune_interval
            if acc_rate < 0.001:
                self.scale *= 0.1
            elif acc_rate < 0.05:
                self.scale *= 0.5
            elif acc_rate < 0.2:
                self.scale *= 0.9
            elif acc_rate > 0.95:
                self.scale *= 10.0
            elif acc_rate > 0.75:
                self.scale *= 2.0
            elif acc_rate > 0.5:
                self.scale *= 1.1
            self.steps_until_tune = self.tune_interval
            self.accepted = 0
        self.steps_until_tune -= 1
        q = q0 + np.random.normal() * self.scale
        return q
    def accept(self, lnp0, lnp):
        if lnp == -np.inf:
            return False
        if -np.inf < np.log(np.random.uniform()) < lnp - lnp0:
            self.accepted += 1
            return True
        return False

class Sampler:
    def __init__(self, q0, lnprob, args=None, scale=None, tune_interval=None, progress_bar=True):
        self.progress_bar = progress_bar
        self.lnprob = lnprob
        self.args = args
        self.walkers = []
        self.q = np.array(q0, dtype=np.float64)
        if scale is None:
            scale = [1 for i in self.q]
        if tune_interval is None:
            tune_interval = [100 for i in self.q]
        for i, _ in enumerate(self.q):
            self.walkers.append(walker(scale[i], tune_interval[i]))
    def run(self, nsamples):
        samples = np.empty((nsamples, self.q.size), dtype=np.float64)
        if self.args is None:
            lnp0 = self.lnprob(self.q)
        else:
            lnp0 = self.lnprob(self.q, *self.args)
        iter_samples = range(nsamples)
        if tqdm is not None and self.progress_bar:
            iter_samples = tqdm(iter_samples)
        for j in iter_samples:
            for i, walker in enumerate(self.walkers):
                q0 = self.q[i]
                self.q[i] = walker.step(q0)
                if self.args is None:
                    lnp = self.lnprob(self.q)
                else:
                    lnp = self.lnprob(self.q, *self.args)
                if walker.accept(lnp0, lnp):
                    lnp0 = lnp
                else:
                    self.q[i] = q0
            samples[j, :] = self.q
        return samples
