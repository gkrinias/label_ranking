import numpy as np
from scipy.special import softmax
from itertools import permutations


def KT(sigma, pi):
  res = 0
  for i in range(len(pi) - 1):
    for j in range(i + 1, len(pi)):
      if sigma[i] < sigma[j] and pi[i] > pi[j] or sigma[i] > sigma[j] and pi[i] < pi[j]:
        res += 1
  return res

class MallowsModel:
  def __init__(self, k, theta=1):
    assert theta >= 0, "Theta must be nonnegative!"

    self.k = k
    self.theta = theta
    self.permutations = None
    self.probabilities = None
    self.__create_distribution()
    self.__rng = np.random.default_rng()
  
  def __KT(self, sigma, pi):
    res = 0
    for i in range(len(pi) - 1):
      for j in range(i + 1, len(pi)):
        if sigma[i] < sigma[j] and pi[i] > pi[j] or sigma[i] > sigma[j] and pi[i] < pi[j]:
          res += 1
    return res
  
  def __create_distribution(self):
    """
    Calculates the probability of each permutation
    """
    self.permutations = list(permutations(range(self.k)))
    self.probabilities = softmax([-self.theta*self.__KT(permutation, range(self.k)) for permutation in self.permutations])

  def __sample_with_default_reference(self):
    """
    Returns a sample of the Mallows distribution with reference permutation [0, 1, ... , k - 1]
    """
    return self.__rng.choice(self.permutations, p=self.probabilities)
  
  def sample(self, reference):
    """
    Returns a sample of the Mallows distribution with given reference permutation
    """
    return np.matmul(np.eye(self.k)[reference], self.__sample_with_default_reference())
  
if __name__ == '__main__':
  mallows = MallowsModel(k=8, theta=6)
  for i in range(10):
    p = mallows.sample(range(8))
    print(KT(p, range(8)))