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
    def __init__(self, q0, lnprob, args=(), scale=None, tune_interval=None, progress_bar=True):
        self.progress_bar = progress_bar
        self.lnprob = lnprob
        self.args = args
        self.walkers = []
        self.q = np.array(q0, dtype=np.float64)
        self.has_blobs = False
        self.lnp0 = self.lnprob(self.q, *self.args)
        try:
            self.lnp0, blob = self.lnp0
            self.has_blobs = True
        except TypeError:
            pass
        if scale is None:
            scale = [1 for i in self.q]
        if tune_interval is None:
            tune_interval = [100 for i in self.q]
        for i in range(len(self.q)):
            self.walkers.append(walker(scale[i], tune_interval[i]))
    def run(self, nsamples, burnin=0):
        samples = np.empty((nsamples, self.q.size), dtype=np.float64)
        blobs = []
        if tqdm is None or not self.progress_bar:
            iter_samples = range(nsamples + burnin)
        else:
            iter_samples = tqdm(range(nsamples + burnin))
        for i in iter_samples:
            if i < burnin:
                self.sample()
            else:
                if self.has_blobs:
                    samples[i-burnin, :], blob = self.sample()
                    blobs.append(blob)
                else:
                    samples[i-burnin, :] = self.sample()
        if self.has_blobs:
            return samples, blobs
        else:
            return samples
    def sample(self):
        for i, walker in enumerate(self.walkers):
            q0 = self.q[i]
            self.q[i] = walker.step(q0)
            lnp = self.lnprob(self.q, *self.args)
            if self.has_blobs:
                lnp, blob = lnp
            if walker.accept(self.lnp0, lnp):
                self.lnp0 = lnp
            else:
                self.q[i] = q0
        if self.has_blobs:
            return self.q, blob
        else:
            return self.q
