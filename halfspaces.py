import numpy as np
from scipy.stats import logistic


# Implementation of the algorithm in the following paper:
# Ilias Diakonikolas, Vasilis Kontonis, Christos Tzamos and Nikos Zarifis. 
# Learning Halfspaces with Massart Noise Under Structured Distributions. 
# In Conference on Learning Theory, pages 1486–1513. PMLR, 2020.


def sign(x):
  """
  Returns 1 if x >= 0 else -1
  """
  return 2*(x >= 0) - 1


def grad_g(x, y, w, sigma):
  """
  Return the gradient value for an iteration of PSGD.
  """
  ywx = y*np.dot(x, w)
  w_norm = np.linalg.norm(w)
  s = logistic(scale=sigma).cdf(-ywx/w_norm)
  return s*(1 - s)*(w*ywx/w_norm**3 - x*y/w_norm)/sigma


def psgd(X, Y, beta, sigma):
  """
  Returns an array of T candidate weight vectors.
  """
  assert len(X) == len(Y)
  W = np.zeros((len(X) + 1, len(X[0])))  # W is a (T + 1) x d matrix
  W[0][0] = 1  # w0 <- e1
  for i in range(1, len(W)):
    W[i] = W[i - 1] - beta*grad_g(X[i - 1], Y[i - 1], W[i - 1], sigma)
    W[i] /= np.linalg.norm(W[i])
  return W[1:]


def best_halfspace_idx(W, X, Y):
  """
  Return the halfspaces that minimizes error among candidates.
  """
  return np.argmin([np.sum(sign(np.sum(w*X, axis=1)) != Y) for w in W])


def halfspace(X, Y, epsilon, delta, eta_max, constants):
  """
  Finds candidate halfspaces applying PSGD and returns the best among them.
  Arguments:
  - d: Feature vector dimension
  - epsilon: misclassification probability
  - delta: confidence parameter
  - 
  """
  d = len(X[0])
  C1, C2, C3, C4 = constants

  T = int(C1 * d * np.log(2/epsilon)**8 / epsilon**4 / (1 - 2*eta_max)**10 * np.log(1/delta))
  beta = C2 * d / np.sqrt(T)
  sigma = C3 * np.sqrt(1 - 2*eta_max) * epsilon / np.log(2/epsilon)**2
  N = int(C4 * np.log(T) / epsilon**2 / (1 - 2*eta_max)**2)

  assert len(X) == len(Y)
  assert len(X) >= T + N, f"At least {T + N} samples needed."

  print("d =", d)
  print("epsilon =", epsilon)
  print("delta =", delta)
  print("T =", T)
  print("eta =", eta_max)
  print("beta =", beta)
  print("sigma =", sigma)
  print("N =", N)
  print()

  W = psgd(X[:T], Y[:T], beta, sigma)
  W = np.concatenate((W, -W))
  return W[best_halfspace_idx(W, X[T:T + N], Y[T:T + N])]


def accuracy(w, X, Y):
  """
  Accuracy of w on X according to ground truth labels Y.
  """
  return np.mean(sign(np.sum(w*X, axis=1)) == Y)