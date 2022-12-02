import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from label_ranking import *
from massart import *

N = 1000000
d = 10  # data dimension
k = 4  # labels
epsilon = 0.3  # upper bound for classification error
delta = 0.3  # 1 - delta is the confidence
eta_max = 0

D = multivariate_normal(mean=np.zeros(d), cov=np.identity(d))  # distribution

W_opt = np.random.rand(k, d)


X_train = D.rvs(size=N)
P_train_ground_true = ground_truth_permutations(X_train, W_opt)
P_train = np.array([addNoise(y, eta_max) for y in P_train_ground_true])

V = pairwiseHalfspaces(
  X=X_train,
  P=P_train,
  epsilon=epsilon,
  delta=delta,
  eta_max=eta_max,
  constants=(1e-5, 1, 1, 10)
)

# Test accuracy of output halfspace
print(np.mean([ranking_accuracy(X_train[i], P_train_ground_true[i], V) for i in range(1000)]))