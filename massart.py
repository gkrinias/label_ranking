import numpy as np


def sign(x):
  """
  Returns 1 if x >= 0 else -1
  """
  return 2*(x >= 0) - 1


def ground_truth_labels(X, w_opt):
  """
  Find ground truth labels given optimal weight vector.
  """
  return np.array([sign(np.dot(x, w_opt)) for x in X])


def flip_label(x, eta_):
  """
  Flip x's ground truth label with probability eta(x) <= eta_max < 1/2.
  """
  return np.random.random() < eta_[0](x, *eta_[1:])


def noisy_labels(X, Y, eta_):
  """
  Find noisy labels of dataset X given ground truth labels and eta function.
  """
  return np.array([-y if flip_label(x, eta_) else y for x, y in zip(X, Y)])


def eta(x, eta_max=0.4, eta_type=1, w=None):
  """
  Method 1: Label flip probability increases exponentially with x's norm
  Method 2: Label flip probability decreases exponentially with x's norm
  Method 3: Label flip probability increases exponentially with the
  distance of x from hyperplane w
  Method 4: Label flip probability increases exponentially with the
  distance of x from hyperplane w
  """
  if eta_type == 1:
    return eta_max*np.exp(-1/max(np.linalg.norm(x), 1e-10))
  if eta_type == 2:
    return eta_max*np.exp(-np.linalg.norm(x))
  if eta_type == 3:
    if w is None: w = np.ones(len(x))
    return eta_max*np.exp(-1/max(np.abs(np.dot(x, w))/np.linalg.norm(w), 1e-10))
  if eta_type == 4:
    if w is None: w = np.ones(len(x))
    return eta_max*np.exp(-np.abs(np.dot(x, w))/np.linalg.norm(w))