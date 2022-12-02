import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from halfspaces import *
from massart import *

from halfspaces_inplace import halfspace as h2


d = 10  # data dimension
epsilon = 0.2  # upper bound for classification error
delta = 0.2  # 1 - delta is the confidence
eta_type = 1
eta_max = 0.2

# We select a random halfspace to be the optimal and 
# ground truth labels are adjusted accordingly.
w_opt = np.random.rand(d)

# We use the standard d-dimensional normal distribution, for which
# U = O(1), R = O(1) and t(epsilon) = O(log(1/epsilon)) (see paper).
D = multivariate_normal(mean=np.zeros(d), cov=np.identity(d))  # distribution

X_train = D.rvs(size=100000)
Y_train_ground_true = ground_truth_labels(X_train, w_opt)
Y_train = noisy_labels(X_train, Y_train_ground_true, (eta, eta_max, eta_type) + (w_opt,))

# Test accuracy of output halfspace
X_test = D.rvs(size=10000)
Y_test = ground_truth_labels(X_test, w_opt)


# w_hat_1 = halfspace(
#   X=X_train,
#   Y=Y_train,
#   epsilon=epsilon,
#   delta=delta,
#   eta_max=0,
#   constants=(1e-1, 1, 1, 10)
# )
# print("Accuracy:", accuracy(w_hat_1, X_test, Y_test))

w_hat_2 = h2(
  d=d,
  epsilon=0.1,
  delta=0.5,
  eta_max=0,
  constants=(0.001, 1, 1, 10),
  distribution=D,
  w_opt=w_opt
)
print("Accuracy:", accuracy(w_hat_2, X_test, Y_test))