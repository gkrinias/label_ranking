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


def kwikSort(V, A):
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

  return [kwikSort(Vl, Al), i, kwikSort(Vr, Ar)]


def flip_pair(eta):
  """
  Decide whether to change the ordering of a pair.
  Change takes place with probability < eta.
  """
  return np.random.random() < eta


def flip_ranking(ranking, eta):
  """
  Flips the given ranking with probability eta.
  This corresponds to adding RCN to each pair.
  """
  assert 0 <= eta < .5
  return np.flip(ranking) if np.random.random() < eta else ranking


def massart_noisy_ranking(ranking, eta, x, W):
  """
  Let l(1), ..., l(k) be the original true ranking corresponding to x.
  We swap labels (l(i), l(i+1)) with probability eta - |(w(l(i)) - w(l(i+1)))x| <= eta
  For i = 0, 2, 4, ...
  """
  assert 0 <= eta < .5
  for i in range(0, len(ranking) if len(ranking) % 2 == 0 else len(ranking) - 1, 2):
    if np.random.random() < eta - 0.1*np.abs(np.dot(W[ranking[i]] - W[ranking[i + 1]], x)):
      ranking[i], ranking[i + 1] = ranking[i + 1], ranking[i]
  return ranking


def flip_ranking_massart(ranking, eta, x):
  """
  Flips the given ranking with probability eta.
  This corresponds to adding RCN to each pair.
  """
  assert 0 <= eta < .5
  return np.flip(ranking) if np.random.random() < eta*(1 - .5*np.abs(x[0])) else ranking




def addNoise(ranking, eta):
  """
  Adds noise to a given ranking.
  The probability to change the ordering in each pair is at most eta.
  """
  pair_orderings = set((j, i) if flip_pair(eta) else (i, j) for (i, j) in combinations(ranking, 2))
  # We need to resolve conflicts, i.e. remove cycles in the corresponding tournament graph
  return np.array(flatten(kwikSort(set(range(len(ranking))), pair_orderings)))

def addNoise2(ranking, eta):
  """
  Flips some specific pairs with probability eta.
  """
  if flip_pair(eta): ranking[0], ranking[1] = ranking[1], ranking[0]
  if flip_pair(eta): ranking[3], ranking[4] = ranking[4], ranking[3]
  return ranking

def addNoise3(ranking, eta):
  """
  Flips some specific pairs with probability eta.
  """
  if flip_pair(eta): ranking[0], ranking[12] = ranking[12], ranking[0]
  if flip_pair(eta): ranking[3], ranking[14] = ranking[14], ranking[3]
  if flip_pair(eta): ranking[6], ranking[7] = ranking[7], ranking[6]
  if flip_pair(eta): ranking[10], ranking[11] = ranking[11], ranking[10]
  return ranking


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