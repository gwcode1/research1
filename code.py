import numpy as np
from scipy.integrate import quad
import pandas as pd
import pymc as pm
from sklearn.linear_model import LinearRegression

class PolynomialBasis:
  def __init__(self, lowerBound=0, upperBound=1, zero=1e-3):
    self.lowerBound = lowerBound
    self.upperBound = upperBound
    self.zero = zero

  def value(self, i, x):
    if i == 0:
      v = 1
    else:
      s = (x - self.lowerBound) / (self.upperBound - self.lowerBound)
      if abs(s) < self.zero:
        v = 0
      elif abs(s - 1) < self.zero:
        v = 1
      else:
        v = s ** i
    return v


class BasisExpansion:
  def __init__(self, basis):
    self.basis = basis

  def project(self, coef, u):
    nsanples = len(u)
    ncoef = len(coef)
    px = np.zeros(nsanples)
    for i in range(nsanples):
      for j in range(ncoef):
          px[i] += coef[j] * self.basis.value(j, u[i])
    return px

  def collocate(self, ndegrees, eps, u, x):
    A = np.zeros((len(u), ndegrees + 1))
    B = np.zeros((len(x), 1))
    for i in range(len(u)):
      B[i] = x[i]
      for j in range(ndegrees + 1):
        A[i, j] = self.basis.value(j, u[i])
    return np.linalg.solve(A.T @ A + eps * np.eye(ndegrees + 1), A.T @ B)


def gen_prior_samples(dataframe, nboot = 250, nbootSamples =  1000):
  model = LinearRegression()
  betas = []
  alphas = []
  for i in range(nboot):
    df_sample = dataframe.sample(nbootSamples, replace = True)
    model.fit(df_sample[['x']],df_sample['y'])
    alphas.append(model.intercept_)
    betas.append(model.coef_[0])
  return alphas, betas


def get_dists(alphas, betas, ndegrees, nproj, nsamples, eps = 1e-3):
  ua = np.sort(np.random.normal(size=len(alphas)))
  minUa = np.min(ua)
  maxUa = np.max(ua)
  basis = PolynomialBasis(minUa, maxUa, eps)
  bexpand = BasisExpansion(basis)
  calpha = bexpand.collocate(ndegrees, 0, ua, alphas)
  cbeta = bexpand.collocate(ndegrees, 0, ua, betas)

  ua = np.random.normal(size=int(nproj))
  palpha = bexpand.project(calpha, ua)
  pbeta = bexpand.project(cbeta, ua)

  with EmpiricalDistribution(palpha) as dist:
    dalpha = dist.empcdfpdf(1, nsamples)
  with EmpiricalDistribution(pbeta) as dist:
    dbeta = dist.empcdfpdf(1, nsamples)
  return dalpha, dbeta

class EmpiricalDistribution:
    def __init__(self, x):
        self._data = {
            "data": x,
            "mean": np.mean(x),
            "var": np.var(x),
            "std": np.std(x),
            "med": np.median(x),
            "cdf": None,
            "pdf": None
        }

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
      pass

    @property
    def data(self):
      return self._data['data']

    @property
    def mean(self):
      return self._data['mean']

    @property
    def var(self):
      return self._data['var']

    @property
    def std(self):
      return self._data['std']

    @property
    def med(self):
      return self._data['med']

    @property
    def cdf(self):
      return self._data['cdf']

    @property
    def pdf(self):
      return self._data['pdf']

    @staticmethod
    def empcdf(X, nsanples):
        xmin = np.min(X)
        xmax = np.max(X)
        xa = np.linspace(xmin, xmax, nsanples + 1)
        F = np.zeros(nsanples + 1)
        for i in range(nsanples + 1):
            F[i] = np.sum(X <= xa[i]) / len(X)
        return { "x": xa, "F": F }

    @staticmethod
    def emppdf(x, F, fac):
        n = len(x)
        m = np.ones(n)
        dx = x[1] - x[0]
        h = fac * dx

        def psi(r):
            return np.exp(-r**2 / 2)

        def dpsi(r):
            return -r * np.exp(-r**2 / 2)

        alfa = 1 / (h * np.sqrt(2 * np.pi))

        def k(p, z):
            return alfa * psi(np.abs(z - p) / h)

        def dk(p, z):
            return alfa * dpsi(np.abs(z - p) / h) * np.sign(z - p) / h

        rho = np.zeros(n)
        for i in range(n):
            for j in range(n):
                rho[i] += m[j] * k(x[i], x[j])

        dy = np.zeros(n)
        for i in range(n):
            s1 = 0
            s2 = 0
            for j in range(n):
                s1 += m[j] * (x[j] - x[i]) * dk(x[i], x[j]) / rho[j]
                s2 += m[j] * (F[j] - F[i]) * dk(x[i], x[j]) / rho[j]
            dy[i] = s2 / s1

        return { "x": x, "f": dy }

    def empcdfpdf(self, fac, nsanples):
        self._data['cdf'] = self.empcdf(self._data['data'], nsanples)
        self._data['pdf'] = self.emppdf(self._data['cdf']['x'], self._data['cdf']['F'], fac)
        return self._data

def pymc_train(x_train, y_train, pdict):
  with pm.Model() as model:
    α = pm.Normal("α", mu=pdict['a_mu'], sigma=pdict['a_sigma'])
    β = pm.Normal("β", mu=pdict['b_mu'], sigma=pdict['b_sigma'])
    ε = pm.HalfNormal("ε", sigma=1.98)

    x = pm.Data('x', x_train, dims='x_data')
    y = pm.Data('y', y_train, dims='y_data')
    μ = pm.Deterministic('μ', α + β * x, dims='x_data')
    
    likelihood = pm.Normal('likelihood', mu=μ, sigma=ε, observed=y, dims='y_data')
    idata = pm.sample(draws=1000, model=model, random_seed=SEED, progressbar=False, idata_kwargs = {'log_likelihood': True})
  return idata
