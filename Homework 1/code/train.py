from itertools import product

import matplotlib.pyplot as plt

from plotdesition import plot_decision_regions


def train(title, classifier, param_grid, X_train, y_train, X_val, y_val, X_test, y_test, plot_desition, log):
    best_param = 0
    best_param_score = 0
    accuracy = []
    param_prod = [dict(zip(param_grid, v)) for v in product(*param_grid.values())]

    print('\n\nTraining ' + title + ': \n')
    for param_dict in param_prod:
        classifier.set_params(**param_dict)

        # Train the estimator on the training set
        classifier.fit(X_train, y_train)

        if plot_desition:
            plot_decision_regions(X_train, y_train, classifier, 'Decision of ' + title + ' (' +
                                  list(param_dict.keys())[0] + ': ' + str(list(param_dict.values())[0]) + ')').show()

        # Evaluate the estimator on the validation set
        val_accuracy_score = classifier.score(X_val, y_val)
        accuracy.append(val_accuracy_score)
        print(str(param_dict) + ', accuracy on validation set: ' + str(val_accuracy_score))

        # Save best parameters
        if val_accuracy_score > best_param_score:
            best_param = param_dict
            best_param_score = val_accuracy_score

    # Plot accuracy for each value of parameter
    plt.figure()
    plt.title(title)
    plt.plot(list(param_grid.values())[0], accuracy)
    plt.xlabel('Value of ' + list(best_param.keys())[0] + ' for ' + title)
    plt.ylabel('Validation Accuracy')
    if log:
        plt.xscale('log')
    plt.show()

    # Train a estimator with the best parameters
    best_classifier = classifier.set_params(**best_param)
    best_classifier.fit(X_train, y_train)

    # Evaluate the estimator on the test set
    test_accuracy_score = best_classifier.score(X_test, y_test)
    print('\nBest ' + str(best_param) + ', accuracy on test set: ' + str(test_accuracy_score))

