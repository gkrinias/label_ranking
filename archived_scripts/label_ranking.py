import numpy as np
from itertools import combinations
from archived_scripts.halfspaces import *


def flatten(S):
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
  return flatten(kwickSort(set(range(len(ranking))), pair_orderings))


def pairwiseHalfspaces(X, P, epsilon, delta, eta_max, constants):
  return {
    (i, j): halfspace(
      X=X,
      Y=[sign(p[j] - p[i]) for p in P],
      epsilon=epsilon/4,
      delta=delta/10/len(P[0])**2,
      eta_max=eta_max,
      constants=constants
    )
    for (i, j) in combinations(range(len(P[0])), 2)
  }


def improperLSF(x, k, V):
  A = set((i, j) if np.dot(V[(i, j)], x) > 0 else (j, i) for (i, j) in combinations(range(k), 2))
  return flatten(kwickSort(set(range(k)), A))


def ranking_accuracy(x, p, V):
  k = len(p)
  p_predicted = np.argsort(improperLSF(x, k, V))
  goodPairs = 0
  for (i, j) in combinations(range(k), 2):
    if (p[i] - p[j])*(p_predicted[i] - p_predicted[j]) > 0: goodPairs += 1
  return 2*goodPairs/k/(k-1)