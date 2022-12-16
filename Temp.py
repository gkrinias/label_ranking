from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.stats import logistic
from itertools import combinations

class LR_PSGD(BaseEstimator, ClassifierMixin):
  def __init__(self, beta, sigma):
    self.NLABELS = None
    self.beta = beta
    self.sigma = sigma
    self.W_ = None

  def __sign(self, x): return 2*(x >= 0) - 1

  def __grad_g(self, x, p, W):
    G = np.empty_like(W)
    for i in range(self.NLABELS):
      t = np.zeros(len(x))
      for j in range(self.NLABELS):
        if i != j:
          wiminuswjx = np.dot(x, W[i] - W[j])
          wiminuswj_norm = np.dot(x, W[i] - W[j])
          s = logistic(scale=self.sigma).cdf((p[i] - p[j])*wiminuswjx/wiminuswj_norm/(self.NLABELS - 1))
          t += s*(1 - s)*(x/wiminuswj_norm - (W[i] - W[j])*wiminuswjx/wiminuswj_norm**3)/self.sigma*(p[i] - p[j])/(self.NLABELS - 1)
      G[i] = t
    return G
    
  def __psgd(self, X, Y):
    W_ = np.zeros((self.NLABELS, len(X[0])))
    for i in range(self.NLABELS): W_[i, i] = 1
    for i in range(len(X)):
      W_ = W_ - self.beta*self.__grad_g(X[i], Y[i], W_)
      for j in range(self.NLABELS): W_[j] /= np.linalg.norm(W_[j])
    return W_

  def fit(self, X, P):
    self.NLABELS = len(P[0])
    self.W_ = self.__psgd(X, P)
    return self

  def predict(self, X):
    return np.array([np.argsort(np.matmul(self.W_, x))[::-1] for x in X])