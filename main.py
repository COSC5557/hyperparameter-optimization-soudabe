import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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


if __name__ == "__main__":
    data_frame = pd.read_csv("winequality-white.csv", sep=";")

    X, y = separate_features_target(data_frame, target_column="quality")

    # Define outer and inner cross-validation strategies
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Random Forest
    rf_params = {
        'n_estimators': (10, 250),
        'max_depth': (1, 50),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
    }
    best_rf_params, rf_score = nested_cv(
        RandomForestClassifier(random_state=42),
        rf_params,
        X, y,
        inner_cv,
        outer_cv,
        'accuracy',
        int_params=['n_estimators', 'max_depth', 'min_samples_split']
    )

    # SVM
    svm_params = {
        'C': (0.1, 100),
        'gamma': (0.001, 10.0)
    }
    best_svm_params, svm_score = nested_cv(SVC(random_state=42), svm_params, X, y, inner_cv, outer_cv, 'accuracy')

    # Gradient Boosting
    gb_params = {
        'n_estimators': (10, 100),
        'learning_rate': (0.01, 0.3),
        'max_depth': (5, 40)
    }
    best_gb_params, gb_score = nested_cv(
        GradientBoostingClassifier(random_state=42),
        gb_params,
        X, y,
        inner_cv,
        outer_cv,
        'accuracy',
        int_params=['n_estimators', 'learning_rate', 'max_depth']
    )

    print("Best Random Forest Parameters:", best_rf_params)
    print("Random Forest Nested CV Score:", rf_score)

    print("Best SVM Parameters:", best_svm_params)
    print("SVM Nested CV Score:", svm_score)

    print("Best Gradient Boosting Parameters:", best_gb_params)
    print("Gradient Boosting Nested CV Score:", gb_score)

    model_names = ["Random Forest", "SVM", "Gradient Boosting"]
    model_scores = [rf_score, svm_score, gb_score]

    plt.figure(figsize=(12, 6))
    plt.bar(model_names, model_scores, color=['blue', 'green', 'red'])
    plt.xlabel('Models')
    plt.ylabel('Nested CV Scores')
    plt.title('Nested Cross-Validation Scores of Different Models')
    plt.xticks(model_names, rotation=45)
    plt.tight_layout()
    plt.show()