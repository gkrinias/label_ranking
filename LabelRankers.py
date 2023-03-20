import numpy as np
from scipy.stats import logistic
from itertools import combinations
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import kendalltau

from sklearn.utils.validation import check_is_fitted


def KTdistance(p, q):
  """
  Returns normalized KT distance between two position arrays.
  """
  assert len(p) == len(q)
  misordered_pairs = 0
  for (i, j) in combinations(range(len(p)), 2):
    if (p[i] - p[j])*(q[i] - q[j]) < 0: misordered_pairs += 1
  return 2*misordered_pairs/len(p)/(len(p) - 1)


def score(P, P_pred):
  return np.mean([1 - KTdistance(p, p_pred) for p, p_pred in zip(P, P_pred)])

def mean_kendall_rank_corr(P_true, P_pred):
  """
  Mean Kendall rank correlation coefficient
  """
  return np.mean([kendalltau(p_true, p_pred)[0] for p_true, p_pred in zip(P_true, P_pred)])



def flatten(S):
  """
  Recursively flatten a list of lists of lists of...
  """
  if not S:
    return S
  if isinstance(S[0], list):
    return flatten(S[0]) + flatten(S[1:])
  return S[:1] + flatten(S[1:])


def kwickSort(V, A):
  """
  Approximation algorithm for constructing a permutation
  from tournament graph (pairwise orderings), so that the
  resultant conflicts are minimized.
  """
  if not V: return []
  Vl = set()
  Vr = set()
  i = np.random.choice(list(V))

  for j in V.difference({i}): Vl.add(j) if (j, i) in A else Vr.add(j)

  Al = set((i, j) for (i, j) in A if i in Vl and j in Vl)
  Ar = set((i, j) for (i, j) in A if i in Vr and j in Vr)

  return [kwickSort(Vl, Al), i, kwickSort(Vr, Ar)]


class LabelwiseDecisionTreeLR(BaseEstimator, ClassifierMixin):
  def __init__(self):
    self.NLABELS = None
    self.regressors = None
  
  def fit(self, X, P):
    self.NLABELS = len(P[0])
    self.regressors = [
      DecisionTreeRegressor().fit(X, P[:, label])
      for label in range(self.NLABELS)
    ]
    return self

  def predict(self, X):
    # check_is_fitted(self)
    Y = np.array([regressor.predict(X) for regressor in self.regressors]).T
    return np.argsort(np.argsort(Y, axis=1), axis=1)


class LabelwiseRandomForestLR(BaseEstimator, ClassifierMixin):
  def __init__(self):
    self.NLABELS = None
    self.regressors = None
  
  def fit(self, X, P):
    self.NLABELS = len(P[0])
    self.regressors = [
      RandomForestRegressor().fit(X, P[:, label])
      for label in range(self.NLABELS)
    ]
    return self

  def predict(self, X):
    # check_is_fitted(self)
    Y = np.array([regressor.predict(X) for regressor in self.regressors]).T
    return np.argsort(np.argsort(Y, axis=1), axis=1)


class PairwiseDecisionTreeLR(BaseEstimator, ClassifierMixin):
  def __init__(self):
    self.NLABELS = None
    self.clfs = None

  def __sign(self, x): return 2*(x >= 0) - 1

  def __create_positions(self, y):
    """
    Takes as argument an array indicating the dominant label for each pair
    of labels and returns the positions of the underlying ranking. 
    """
    A = set(
      (i, j) if y[k] > 0 else (j, i) 
      for k, (i, j) in enumerate(combinations(range(self.NLABELS), 2))
    )
    return np.argsort(flatten(kwickSort(set(range(self.NLABELS)), A)))

  def fit(self, X, P):
    self.NLABELS = len(P[0])
    self.clfs = [
      DecisionTreeClassifier().fit(
        X,
        self.__sign(P[:, j] - P[:, i])
      )
      for (i, j) in combinations(range(self.NLABELS), 2)
    ]
    return self

  def predict(self, X):
    Y = np.array([clf.predict(X) for clf in self.clfs]).T
    return np.array([self.__create_positions(y) for y in Y])


class PairwiseRandomForestLR(BaseEstimator, ClassifierMixin):
  def __init__(self):
    self.NLABELS = None
    self.clfs = None

  def __sign(self, x): return 2*(x >= 0) - 1

  def __create_positions(self, y):
    """
    Takes as argument an array indicating the dominant label for each pair
    of labels and returns the positions of the underlying ranking. 
    """
    A = set(
      (i, j) if y[k] > 0 else (j, i) 
      for k, (i, j) in enumerate(combinations(range(self.NLABELS), 2))
    )
    return np.argsort(flatten(kwickSort(set(range(self.NLABELS)), A)))

  def fit(self, X, P):
    self.NLABELS = len(P[0])
    self.clfs = [
      RandomForestClassifier().fit(
        X,
        self.__sign(P[:, j] - P[:, i])
      )
      for (i, j) in combinations(range(self.NLABELS), 2)
    ]
    return self

  def predict(self, X):
    Y = np.array([clf.predict(X) for clf in self.clfs]).T
    return np.array([self.__create_positions(y) for y in Y])


class PairwiseHalfspaceLR(BaseEstimator, ClassifierMixin):
  def __init__(self, beta, sigma, split):
    self.NLABELS = None
    self.beta = beta
    self.sigma = sigma
    self.split = split
    self.V = None

  def __sign(self, x): return 2*(x >= 0) - 1

  def __create_positions(self, y):
    """
    Takes as argument an array indicating the dominant label for each pair
    of labels and returns the positions of the underlying ranking. 
    """
    A = set(
      (i, j) if y[k] > 0 else (j, i) 
      for k, (i, j) in enumerate(combinations(range(self.NLABELS), 2))
    )
    return np.argsort(flatten(kwickSort(set(range(self.NLABELS)), A)))

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
    T = int(len(X)*self.split)
    W = self.__psgd(X[:T], Y[:T])
    W = np.concatenate((W, -W))
    return W[self.__best_halfspace_idx(W, X[T:], Y[T:])]

  def fit(self, X, P):
    self.NLABELS = len(P[0])
    self.V = np.array([
      self.__halfspace(X, self.__sign(P[:, j] - P[:, i]))
      for (i, j) in combinations(range(self.NLABELS), 2)
    ])
    return self

  def predict(self, X):
    Y = np.matmul(self.V, X.T).T
    return np.array([self.__create_positions(y) for y in Y])