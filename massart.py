import numpy as np
from itertools import combinations


def sign(x):
  """
  Returns 1 if x >= 0 else -1
  """
  return 2*(x >= 0) - 1


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


def flip_pair(eta):
  """
  Decide whether to change the ordering of a pair.
  Change takes place with probability < eta.
  """
  return np.random.random() < eta


def addNoise(ranking, eta):
  """
  Adds noise to a given ranking.
  The probability to change the ordering in each pair is at most eta.
  """
  pair_orderings = set((j, i) if flip_pair(eta) else (i, j) for (i, j) in combinations(ranking, 2))
  # We need to resolve conflicts, i.e. remove cycles in the corresponding tournament graph
  return np.array(flatten(kwickSort(set(range(len(ranking))), pair_orderings)))


def ground_truth_labels(X, w_opt):
  """
  Find ground truth labels given optimal weight vector.
  """
  return np.array([sign(np.dot(x, w_opt)) for x in X])


def ground_truth_permutations(X, W_opt):
  """
  Find ground truth labels given optimal weight vector.
  """
  return np.array([np.argsort(np.matmul(W_opt, x))[::-1] for x in X])


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