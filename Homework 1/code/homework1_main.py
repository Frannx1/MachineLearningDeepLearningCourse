import os
import sys
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
import numpy as np
from sklearn.svm import SVC

from classifiers import dict_classifiers
from plotdesition import plot_decision_regions
from train import train

test_size = 0.3
val_size = 0.2 / (1 - test_size)

# 1.
raw_data = datasets.load_wine()

# 2.
data_projected = np.array([row[0:2] for row in raw_data['data']])

# 3.a
X_train_merge, X_test, y_train_merge, y_test = \
    train_test_split(data_projected, raw_data['target'], test_size=test_size, random_state=1234)

# 3.b
X_train, X_val, y_train, y_val = \
    train_test_split(X_train_merge, y_train_merge, test_size=val_size, random_state=4)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"     # Also affect subprocesses

# 4-14
plot_desition = True
for key, classifier in dict_classifiers.items():
    log = True
    if key == 'Nearest Neighbors':
        log = False
    train(key, classifier['classifier'], classifier['params'],
          X_train, y_train, X_val, y_val, X_test, y_test, plot_desition, log)

# 15
classifier = SVC()
C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
gamma_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

param_grid = dict(gamma=gamma_range, C=C_range)
best_param = {}
best_param_score = 0
grid_accuracy = pd.DataFrame(columns=C_range,
                             index=gamma_range)

print('Grid search for RVF Kernel')
for k, param_dict in enumerate(ParameterGrid(param_grid)):
    classifier.set_params(**param_dict)
    classifier.fit(X_train, y_train)

    val_accuracy_score = classifier.score(X_val, y_val)
    grid_accuracy._set_value(param_dict['gamma'], param_dict['C'], val_accuracy_score)

    if val_accuracy_score > best_param_score:
        best_param = param_dict

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(np.array(grid_accuracy, dtype=float))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(grid_accuracy)

print('\nBest parameters: C = ' + str(best_param['C']) + ', gamma = ' + str(best_param['gamma']))

classifier.set_params(**best_param)
classifier.fit(X_train, y_train)
accuracy_test = classifier.score(X_test, y_test)

print('Accuracy on test set: ' + str(accuracy_test))
plot_decision_regions(X_train, y_train, classifier, 'Decision of RBF Kernel (C = ' + str(best_param['C']) +
                      ', gamma = ' + str(best_param['gamma']) + ')').show()

# 17
print('\nGrid search for RVF Kernel with 5-fold')
grid = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1, refit=True)
estimator = grid.fit(X_train_merge, y_train_merge)
print('\nBest parameters: C = ' + str(estimator.best_params_['C']) + ', gamma = ' + str(estimator.best_params_['gamma']))


scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))
plt.figure()
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()

plot_decision_regions(X_train, y_train, grid.best_estimator_, 'Decision of RBF Kernel (C = ' + str(best_param['C']) +
                      ', gamma = ' + str(best_param['gamma']) + ')').show()

accuracy_validation = estimator.best_score_
print('Accuracy on cross validation: ' + str(accuracy_validation))

# 18
accuracy_test = estimator.score(X_test, y_test)
print('Accuracy on test set: ' + str(accuracy_test))

