import pickle
from LabelRanking import *
import copy
from tqdm import tqdm

def run_algorithm(
  classifier,
  classifier_name,
  dimensionality='SFN',
  noise_type='additive',
  noise_measurement='alpha'
  ):
  with open(f'datasets/{dimensionality}/X_train.pickle', 'rb') as handle: X_train = pickle.load(handle)
  with open(f'datasets/{dimensionality}/X_test.pickle', 'rb') as handle: X_test = pickle.load(handle)
  with open(f'datasets/{dimensionality}/P_test.pickle', 'rb') as handle: P_test = pickle.load(handle)

  clfs = []
  measurement_params = []

  for i in range(50):
    with open(f'datasets/{dimensionality}/{noise_type}/{noise_measurement}_{i}.pickle', 'rb') as handle:
      data = pickle.load(handle)
    # print(data[noise_measurement])
    measurement_params.append(data[noise_measurement])
    P_train_noisy = data['P_train_noisy']
    clf = copy.deepcopy(classifier).fit(X_train, P_train_noisy)
    clfs.append(clf)
  
  clf_preds = [clf.predict(X_test) for clf in clfs]
  clf_KT_corr = [mean_KTcorrelation(P_test, clf_pred) for clf_pred in clf_preds]

  res = [x for _, x in sorted(zip(measurement_params, clf_KT_corr))]

  with open(f'results/{dimensionality}/{noise_type}/{noise_measurement}_{classifier_name}.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
  print(classifier_name, res)

hh_params = {
  'SFN': dict(beta=0.005, sigma=0.1),
  'MFN': dict(beta=0.0008, sigma=0.15)
}

dt_params = {
  'SFN': dict(max_depth=10),
  'MFN': dict(max_depth=7)
}

rf_params = {
  'SFN': dict(max_features='log2', n_jobs=-1),
  'MFN': dict(max_features='log2', n_jobs=-1)
}

if __name__ == '__main__':
  for dimensionality in tqdm(['SFN', 'MFN']):
    for noise_type in tqdm(['additive', 'mallows']):
      for noise_measurement in tqdm(['alpha', 'beta']):
        run_algorithm(
          classifier=LabelwiseLabelRanking('Linear', dict(n_jobs=-1)),
          classifier_name='labelwise_lr',
          dimensionality=dimensionality,
          noise_type=noise_type,
          noise_measurement=noise_measurement
        )

        run_algorithm(
          classifier=LabelwiseLabelRanking('Decision Tree', dt_params[dimensionality]),
          classifier_name='labelwise_dt',
          dimensionality=dimensionality,
          noise_type=noise_type,
          noise_measurement=noise_measurement
        )

        run_algorithm(
          classifier=PairwiseLabelRanking('Decision Tree', dt_params[dimensionality], aggregation='tournament'),
          classifier_name='pairwise_dt',
          dimensionality=dimensionality,
          noise_type=noise_type,
          noise_measurement=noise_measurement
        )

        run_algorithm(
          classifier=LabelwiseLabelRanking('Random Forest', rf_params[dimensionality]),
          classifier_name='labelwise_rf',
          dimensionality=dimensionality,
          noise_type=noise_type,
          noise_measurement=noise_measurement
        )

        run_algorithm(
          classifier=PairwiseLabelRanking('Random Forest', rf_params[dimensionality], aggregation='tournament'),
          classifier_name='pairwise_rf',
          dimensionality=dimensionality,
          noise_type=noise_type,
          noise_measurement=noise_measurement
        )

        run_algorithm(
          classifier=PairwiseLabelRanking('Homogeneous Halfspace', hh_params[dimensionality], aggregation='tournament'),
          classifier_name='pairwise_hh',
          dimensionality=dimensionality,
          noise_type=noise_type,
          noise_measurement=noise_measurement
        )