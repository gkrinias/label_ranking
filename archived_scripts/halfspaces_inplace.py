import numpy as np
from scipy.stats import logistic

from massart import *


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


def halfspace(d, epsilon, delta, eta_max, constants, distribution, w_opt):
  """
  Returns an array of T candidate weight vectors.
  """
  C1, C2, C3, C4 = constants
  T = int(C1 * d * np.log(2/epsilon)**8 / epsilon**4 / (1 - 2*eta_max)**10 * np.log(1/delta))
  beta = C2 * d / np.sqrt(T)
  sigma = C3 * np.sqrt(1 - 2*eta_max) * epsilon / np.log(2/epsilon)**2
  N = int(C4 * np.log(T) / epsilon**2 / (1 - 2*eta_max)**2)

  print("d =", d)
  print("epsilon =", epsilon)
  print("delta =", delta)
  print("T =", T)
  print("eta =", eta_max)
  print("beta =", beta)
  print("sigma =", sigma)
  print("N =", N)
  print()

  X = distribution.rvs(size=N)
  Y = ground_truth_labels(X=X, w_opt=w_opt)  # TODO: change with noisy ones
  w_prev = np.zeros(d)  # w0 <- e1
  w_prev[0] = 1
  min_error = N

  for _ in range(T):
    x = distribution.rvs()
    y = ground_truth_labels(X=[x], w_opt=w_opt)  # TODO: change with noisy one
    w = w_prev - beta*grad_g(x, y, w_prev, sigma)
    w /= np.linalg.norm(w)
    w_plus_error = np.sum(sign(np.sum(w*X, axis=1)) != Y)
    w_minus_error = np.sum(sign(np.sum(-w*X, axis=1)) != Y)
    if w_plus_error < min_error:
      w_best = w
      min_error = w_plus_error
    if w_minus_error < min_error:
      w_best = -w
      min_error = w_minus_error
    w_prev = w
  
  return w_best


def accuracy(w, X, Y):
  """
  Accuracy of w on X according to ground truth labels Y.
  """
  return np.mean(sign(np.sum(w*X, axis=1)) == Y)