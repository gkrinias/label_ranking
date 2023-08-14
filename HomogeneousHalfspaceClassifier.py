import numpy as np
from scipy.stats import logistic
from sklearn.base import BaseEstimator, ClassifierMixin


class HomogeneousHalfspaceClassifier(BaseEstimator, ClassifierMixin):
  """
  Homogeneous Halfspace learner that satisfies the PAC guarantee
  for distributions with Massart noise and bounded marginals
  """
  def __init__(self, beta, sigma, split=0.3):
    self.NFEATURES = None
    self.w = None
    self.beta = beta
    self.sigma = sigma
    self.split = split

    assert 0 <= split <= 1
    assert beta > 0
    assert sigma > 0

  def __sign(self, x): return 2*(x >= 0) - 1

  def __grad_g(self, x, y, w):
    ywx = y*np.dot(x, w)
    w_norm = np.linalg.norm(w)
    s = logistic(scale=self.sigma).cdf(-ywx/w_norm)
    return s*(1 - s)*(w*ywx/w_norm**3 - x*y/w_norm)/self.sigma
    
  def __psgd(self, X, Y):
    W = np.zeros((len(X) + 1, len(X[0])))
    W[0][0] = 1
    for i in range(1, len(W)):
      W[i] = W[i - 1] - self.beta*self.__grad_g(X[i - 1], Y[i - 1], W[i - 1])
      W[i] /= np.linalg.norm(W[i])
    return W[1:]
    
  def __best_halfspace_idx(self, W, X, Y):
    return np.argmin([np.sum(self.__sign(np.sum(w*X, axis=1)) != Y) for w in W])

  def __halfspace(self, X, Y):
    W = self.__psgd(X, Y)
    W = np.concatenate((W, -W))
    evaluation_samples = int(np.ceil(self.split*len(X)))
    return W[self.__best_halfspace_idx(W, X[:evaluation_samples], Y[:evaluation_samples])]

  def fit(self, X, Y):
    self.NFEATURES = X.shape[1]
    self.w = self.__halfspace(X, Y)
    return self

  def predict(self, X):
    return self.__sign(np.sum(self.w*X, axis=1))