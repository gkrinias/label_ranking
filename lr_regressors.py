import numpy as np
from itertools import combinations
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def KTdistance(p, q, format='positions'):
  """
  Returns normalized KT distance between two position arrays
  """
  assert len(p) == len(q)
  assert format == 'ranking' or format == 'positions'
  if format == 'ranking':
    p = np.argsort(p)
    q = np.argsort(q)

  misordered_pairs = 0
  for (i, j) in combinations(range(len(p)), 2):
    if (p[i] - p[j])*(q[i] - q[j]) < 0: misordered_pairs += 1
  return 2*misordered_pairs/len(p)/(len(p) - 1)


def DecisionTreeScore(X_train, Y_train, X_test, Y_test, format='positions'):
  """
  Returns average normalized KT distance measured on test data
  after training an sklearn DecisionTreeRegressor on train data
  """
  assert format == 'ranking' or format == 'positions'
  if format == 'ranking':
    Y_train = np.argsort(Y_train)
    Y_test = np.argsort(Y_test)

  k = Y_train.shape[1]  # number of labels
  regressors = [DecisionTreeRegressor().fit(X_train, Y_train[:, label]) for label in range(k)]
  Y_pred = np.array([regressors[label].predict(X_test) for label in range(k)]).T
  R_pred = np.argsort(Y_pred, axis=1)
  P_pred = np.argsort(R_pred, axis=1)

  return np.mean([KTdistance(p_pred, y_test) for (p_pred, y_test) in zip(P_pred, Y_test)])


def RandomForestScore(X_train, Y_train, X_test, Y_test, format='positions'):
  """
  Returns average normalized KT distance measured on test data
  after training an sklearn RandomForestRegressor on train data
  """
  assert format == 'ranking' or format == 'positions'
  if format == 'ranking':
    Y_train = np.argsort(Y_train)
    Y_test = np.argsort(Y_test)

  k = Y_train.shape[1]  # number of labels
  regressors = [RandomForestRegressor().fit(X_train, Y_train[:, label]) for label in range(k)]
  Y_pred = np.array([regressors[label].predict(X_test) for label in range(k)]).T
  R_pred = np.argsort(Y_pred, axis=1)
  P_pred = np.argsort(R_pred, axis=1)

  return np.mean([KTdistance(p_pred, y_test) for (p_pred, y_test) in zip(P_pred, Y_test)])