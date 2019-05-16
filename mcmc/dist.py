import numpy as np

def draw_samples(population, sample_nb=1):
    '''
    Draw samples from a population.

    Parameters
    ----------
    population: numpy array
        Population to sample from.

    sample_nb: int
        Number of samples.

    Returns
    -------
    samples: numpy array
        The samples.
    '''

    samples = np.random.choice(population, sample_nb)
    return samples

def sample_from_cdf(cdf, sample_nb=1):
    '''
    Draw samples from a Cumulative Distribution Function (CDF).

    Parameters
    ----------
    cdf: numpy array
        CDF to sample from.
    sample_nb: int
        Number of samples.

    Returns
    -------
    samples: numpy array
        The samples.
    '''

    x, y = cdf
    ys = np.random.random_sample((sample_nb,))
    samples = xs = np.interp(ys, y, x)
    return samples

def pdf_from_cdf(cdf):
    '''
    Compute Probability Density Function (PDF) from Cumulative Distribution Function (CDF).

    Parameters
    ----------
    cdf:
        The CDF.

    Returns
    -------
    pdf:
        The PDF.
    '''

    x, y = cdf
    pdf = x, np.diff(y, prepend=0)
    return pdf

def cdf_from_samples(samples, nb=100):
    n = len(samples)
    x = np.sort(samples)
    y = np.arange(1, n+1) / n
    a = x[0]
    b = x[-1]
    _x = np.linspace(a, b, nb)
    _y = np.interp(_x, x, y)
    cdf = _x, _y
    return cdf

def cdf_from_pdf(pdf):
    '''
    Compute Cumulative Distribution Function (CDF) from Probability Density Function (PDF).

    Parameters
    ----------
    pdf:
        The PDF.

    Returns
    -------
    cdf:
        The CDF.
    '''

    x, y = pdf
    p = np.cumsum(y)
    cdf = x, p / p.max()
    return cdf

def uniform_pdf(a, b, nb=100):
    xy = np.empty((2, nb))
    xy[0] = np.linspace(a, b, nb)
    xy[1, :] = 1
    xy[1] /= np.trapz(xy[1], x=xy[0])
    return xy

def logp_from_pdf(pdf, x):
    _x, y = pdf
    logp = np.interp(x, _x, y, 0, 0)
    return logp
