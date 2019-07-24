import numpy as np
from scipy import stats

def sample_from_population(population, nb=1):
    '''
    Draw samples from a population.

    Parameters
    ----------
    population: numpy array
        Population to sample from.

    nb: int
        Number of samples to draw.

    Returns
    -------
    samples: numpy array
        The samples.
    '''

    samples = np.random.choice(population, nb)
    return samples

def sample_from_cdf(cdf, nb=1):
    '''
    Draw samples from a Cumulative Distribution Function (CDF).

    Parameters
    ----------
    cdf: numpy array
        CDF to sample from.
    nb: int
        Number of samples to draw.

    Returns
    -------
    samples: numpy array
        The samples.
    '''

    x, y = cdf
    ys = np.random.random_sample((nb,))
    samples = np.interp(ys, y, x)
    return samples

def pdf_from_cdf(cdf):
    '''
    Compute the Probability Density Function (PDF) from Cumulative Distribution
    Function (CDF).

    Parameters
    ----------
    cdf: 2-dimension array
        The CDF.

    Returns
    -------
    pdf: 2-dimension array
        The PDF.
    '''

    x, y = cdf
    pdf = np.empty_like(cdf)
    pdf[0] = x
    pdf[1] = np.diff(y, prepend=0)
    return pdf

def pdf_from_samples(samples, nb=100, kde=False, a=np.inf, b=-np.inf):
    '''
    Compute the Probability Density Function (PDF) from a population.

    Parameters
    ----------
    samples: numpy array
        The population.
    nb: int
        The number of points in the PDF.
    kde: bool
        If True, compute the PDF as a Kernel Density Estimation.
        Otherwise, compute the PDF from the CDF.
    a: float
        The lower bound of the value range. If not provided, it is the minimum
        value.
    b: float
        The upper bound of the value range. If not provided, it is the maximum
        value.

    Returns
    -------
    pdf: 2-dimension array
        The PDF.
    '''

    if kde:
        samples = samples[np.isfinite(samples)]
        smin, smax = np.min(samples), np.max(samples)
        if smin == smax:
            smin *= 0.99
            smax *= 1.01
            samples[:2] = [smin, smax]
        if a < smin:
            smin = a
        if b > smax:
            smax = b
        xy = np.empty((2, nb))
        xy[0] = np.linspace(smin, smax, nb)
        xy[1] = stats.gaussian_kde(samples)(xy[0])
        return xy
    else:
        cdf = cdf_from_samples(samples, nb=nb, a=a, b=b)
        pdf = pdf_from_cdf(cdf)
        pdf[1] /= np.trapz(pdf[1], x=pdf[0])
        return pdf

def cdf_from_samples(samples, nb=100, a=np.inf, b=-np.inf):
    '''
    Compute the Cumulative Distribution Function (CDF) from a population.

    Parameters
    ----------
    samples: numpy array
        The population.
    nb: int
        The number of points in the CDF.
    a: float
        The lower bound of the value range. If not provided, it is the minimum
        value.
    b: float
        The upper bound of the value range. If not provided, it is the maximum
        value.

    Returns
    -------
    cdf: 2-dimension array
        The CDF.
    '''

    n = len(samples)
    x = np.sort(samples)
    y = np.arange(1, n+1) / n
    smin = min(a, x[0])
    smax = max(b, x[-1])
    _x = np.linspace(smin, smax, nb)
    _y = np.interp(_x, x, y)
    cdf = np.empty((2, nb))
    cdf[0] = _x
    cdf[1] = _y
    return cdf

def cdf_from_pdf(pdf):
    '''
    Compute Cumulative Distribution Function (CDF) from Probability Density
    Function (PDF).

    Parameters
    ----------
    pdf: 2-dimension array
        The PDF.

    Returns
    -------
    cdf: 2-dimension array
        The CDF.
    '''

    x, y = pdf
    p = np.cumsum(y)
    cdf = np.empty_like(pdf)
    cdf[0] = x
    cdf[1] = p / p.max()
    return cdf

def uniform_pdf(a, b, nb=100):
    '''
    Create a uniform Probability Distribution Function (PDF).

    Parameters
    ----------
    a: float
        Lower boundary of the interval.
    b: float
        Upper boundary of the interval.

    Returns
    -------
    xy: 2-dimension array
        The PDF.
    '''

    xy = np.empty((2, nb))
    xy[0] = np.linspace(a, b, nb)
    xy[1] = 1
    xy[1] /= np.trapz(xy[1], x=xy[0])
    return xy

def logp_from_pdf(pdf, x, outside_max_factor=0.01):
    '''
    Compute the log-probability given a Probability Distribution Function
    (PDF).

    Parameters
    ----------
    pdf: 2-dimension array
        The PDF.
    x: float or numpy array
        The point(s) where to compute the log-probability.
    outside_max_factor: float
        The probability outside the range of values in the PDF is equal to the
        maximum probability multiplied by this factor (it can be good to have
        non-null probability).

    Returns
    -------
    logp: float or numpy array
        The log-probability.
    '''

    _x, y = pdf
    p_outside = y.max() * outside_max_factor
    logp = np.log(np.interp(x, _x, y, p_outside, p_outside))
    return logp
