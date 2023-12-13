import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from bayes_opt import BayesianOptimization


def split_data(df, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state)


def separate_features_target(df, target_column):
    features = df.drop([target_column], axis=1)
    target = df[target_column].copy()
    return features, target


def nested_cv(estimator, param_bounds, data, target, inner_cv, outer_cv, scoring, int_params=[]):
    def hyperopt_fn(**params):
        for param in int_params:
            if param in params:
                params[param] = int(params[param])

        estimator.set_params(**params)
        scores = cross_val_score(estimator, data, target, scoring=scoring, cv=inner_cv)
        return scores.mean()

    optimizer = BayesianOptimization(f=hyperopt_fn, pbounds=param_bounds, random_state=42)
    optimizer.maximize(init_points=5, n_iter=10)

    best_params = optimizer.max['params']
    for param in int_params:
        if param in best_params:
            best_params[param] = int(best_params[param])

    estimator.set_params(**best_params)

    outer_scores = cross_val_score(estimator, data, target, scoring=scoring, cv=outer_cv)
    return best_params, np.mean(outer_scores)


def default_nested_cv(estimator, data, target, cv, scoring):
    scores = cross_val_score(estimator, data, target, cv=cv, scoring=scoring)
    return np.mean(scores)


if __name__ == "__main__":
    data_frame = pd.read_csv("winequality-white.csv", sep=";")
    X, y = separate_features_target(data_frame, target_column="quality")

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Random Forest
    rf_default_score = default_nested_cv(RandomForestClassifier(random_state=42), X, y, outer_cv, 'accuracy')
    rf_params = {
        'n_estimators': (50, 500),
        'max_depth': (3, 100),
        'min_samples_split': (2, 50),
        'max_features': (0.1, 1.0)
    }
    best_rf_params, rf_optimized_score = nested_cv(
        RandomForestClassifier(random_state=42),
        rf_params,
        X, y,
        inner_cv,
        outer_cv,
        'accuracy',
        int_params=['n_estimators', 'max_depth', 'min_samples_split']
    )

    # SVM
    svm_default_score = default_nested_cv(SVC(random_state=42), X, y, outer_cv, 'accuracy')
    svm_params = {
        'C': (0.1, 100),
        'gamma': (0.001, 10.0)
    }
    best_svm_params, svm_optimized_score = nested_cv(SVC(random_state=42), svm_params, X, y, inner_cv, outer_cv, 'accuracy')

    # AdaBoost
    ab_default_score = default_nested_cv(AdaBoostClassifier(random_state=42), X, y, outer_cv, 'accuracy')

    ab_params = {
        'n_estimators': (50, 500),
        'learning_rate': (0.01, 2)
    }

    best_ab_params, ab_optimized_score = nested_cv(
        AdaBoostClassifier(random_state=42),
        ab_params,
        X, y,
        inner_cv,
        outer_cv,
        'accuracy',
        int_params=['n_estimators']
    )

    print("Random Forest Default Nested CV Score:", rf_default_score)
    print("Best Random Forest Parameters:", best_rf_params)
    print("Random Forest Nested CV Score:", rf_optimized_score)

    print("SVM Default Nested CV Score:", svm_default_score)
    print("Best SVM Parameters:", best_svm_params)
    print("SVM Nested CV Score:", svm_optimized_score)

    print("AdaBoost Default Nested CV Score:", ab_default_score)
    print("Best AdaBoost Parameters:", best_ab_params)
    print("AdaBoost Nested CV Score:", ab_optimized_score)

    # Plotting
    model_names = ["Random Forest", "SVM", "AdaBoost"]
    default_scores = [rf_default_score, svm_default_score, ab_default_score]
    optimized_scores = [rf_optimized_score, svm_optimized_score, ab_optimized_score]

    bar_width = 0.35
    index = np.arange(len(model_names))

    plt.figure(figsize=(12, 6))
    bar1 = plt.bar(index, default_scores, bar_width, label='Default Parameters', color='b')
    bar2 = plt.bar(index + bar_width, optimized_scores, bar_width, label='Optimized Parameters', color='g')

    plt.xlabel('Models')
    plt.ylabel('Nested CV Scores')
    plt.title('Comparison of Default and Optimized Nested CV Scores')
    plt.xticks(index + bar_width / 2, model_names)
    plt.legend()

    plt.tight_layout()
    plt.show()
