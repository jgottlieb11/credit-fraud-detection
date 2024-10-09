import pickle
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from imblearn.over_sampling import ADASYN

print('Loading creditcard.csv')
data = pd.read_csv("data/creditcard.csv")

fraud = shuffle(data[data.Class == 1])
normal = shuffle(data[data.Class == 0])

X_train = fraud.sample(frac=0.8)
X_train = pd.concat([X_train, normal.sample(frac=0.8)], axis=0)

ada = ADASYN(n_jobs=3)

print('Starting ADASYN...')

data_resampled, data_labels_resampled = ada.fit_resample(
    np.array(X_train.loc[:, X_train.columns != 'Class']),
    np.array(X_train['Class'])
)

print('Pickling...')

with open('pickle/train_data_resampled.pkl', 'wb+') as f:
    pickle.dump(data_resampled, f)

with open('pickle/train_data_labels_resampled.pkl', 'wb+') as f:
    pickle.dump(data_labels_resampled, f)

print('Done!')
