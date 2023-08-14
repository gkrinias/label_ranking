import numpy as np
from itertools import combinations
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from HomogeneousHalfspaceClassifier import HomogeneousHalfspaceClassifier
from Graph import tournament2ranking


def KTdistance(sigma, pi):
  res = 0
  for i in range(len(pi) - 1):
    for j in range(i + 1, len(pi)):
      if sigma[i] < sigma[j] and pi[i] > pi[j] or sigma[i] > sigma[j] and pi[i] < pi[j]:
        res += 1
  return res


def mean_KTdistance(P_true, P_pred):
  return np.mean([KTdistance(p_true, p_pred) for p_true, p_pred in zip(P_true, P_pred)])


def mean_KTcorrelation(P_true, P_pred):
  assert P_true.shape == P_pred.shape
  return 1 - 4*mean_KTdistance(P_true, P_pred)/P_true.shape[1]/(P_true.shape[1] - 1)


def LinearSortingFunction(W, X, N=None):
  """
  X: nxd data matrix
  W: kxd matrix
  N: nxk noise matrix
  """
  if N is None: N = np.zeros((X.shape[0], W.shape[0]))
  return np.flip(np.argsort(np.matmul(X, W.T) + N), axis=1)


class LabelwiseLabelRanking(BaseEstimator, ClassifierMixin):
  def __init__(self, regressor_name, params):
    self.NFEATURES = None
    self.NLABELS = None
    self.regressors = None
    self.regressor_name = regressor_name
    self.params = params
  
  def fit(self, X, P):
    self.NFEATURES = X.shape[1]
    self.NLABELS = P.shape[1]

    if self.regressor_name == 'Linear':
      self.regressors = [LinearRegression(**self.params) for _ in range(self.NLABELS)]
    if self.regressor_name == 'Decision Tree':
      self.regressors = [DecisionTreeRegressor(**self.params) for _ in range(self.NLABELS)]
    if self.regressor_name == 'Random Forest':
      self.regressors = [RandomForestRegressor(**self.params) for _ in range(self.NLABELS)]
    
    for label in range(self.NLABELS):
      self.regressors[label].fit(X, P[:, label]/self.NLABELS)

    return self

  def predict(self, X):
    Y = np.array([regressor.predict(X) for regressor in self.regressors]).T
    return np.argsort(np.argsort(Y, axis=1), axis=1)


class PairwiseLabelRanking(BaseEstimator, ClassifierMixin):
  def __init__(self, classifier_name, params, aggregation):
    self.NFEATURES = None
    self.NLABELS = None
    self.clfs = None
    self.classifier_name = classifier_name
    self.params = params
    self.aggregation = aggregation

  def __sign(self, x): return 2*(x >= 0) - 1

  def __votingAggregation(self, y):
    """
    Takes as argument an array indicating the dominant label for each pair
    of labels and returns the positions of the underlying ranking based on
    the number of wins of each label in all pairwise duels 
    """
    s = np.zeros(self.NLABELS)
    for k, (i, j) in enumerate(combinations(range(self.NLABELS), 2)):
      if y[k] == 1: s[i] += 1
      else: s[j] += 1

    return np.argsort(np.argsort(s)[::-1])

  def __tournamentAggregation(self, y):
    """
    Takes as argument an array indicating the dominant label for each pair
    of labels and returns the positions of the underlying ranking based on
    the topological ordering of the underlying tournament after breaking its cycles
    """
    A = set(
      (i, j) if y[k] > 0 else (j, i) 
      for k, (i, j) in enumerate(combinations(range(self.NLABELS), 2))
    )

    return np.argsort(tournament2ranking(self.NLABELS, A))

  def fit(self, X, P):
    self.NFEATURES = X.shape[1]
    self.NLABELS = P.shape[1]

    if self.classifier_name == 'Homogeneous Halfspace':
      self.clfs = [
        HomogeneousHalfspaceClassifier(**self.params).fit(
          X,
          self.__sign(P[:, j] - P[:, i])
        )
        for (i, j) in combinations(range(self.NLABELS), 2)
      ]

    if self.classifier_name == 'Decision Tree':
      self.clfs = [
        DecisionTreeClassifier(**self.params).fit(
          X,
          self.__sign(P[:, j] - P[:, i])
        )
        for (i, j) in combinations(range(self.NLABELS), 2)
      ]

    if self.classifier_name == 'Random Forest':
      self.clfs = [
        RandomForestClassifier(**self.params).fit(
          X,
          self.__sign(P[:, j] - P[:, i])
        )
        for (i, j) in combinations(range(self.NLABELS), 2)
      ]
    
    return self

  def predict(self, X):
    Y = np.array([clf.predict(X) for clf in self.clfs]).T
    if self.aggregation == 'tournament':
      return np.array([self.__tournamentAggregation(y) for y in Y])
    if self.aggregation == 'voting':
      return np.array([self.__votingAggregation(y) for y in Y])