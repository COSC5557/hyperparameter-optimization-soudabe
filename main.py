import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from bayes_opt import BayesianOptimization


def split_data(df, test_size=0.2, random_state=42):
    # Split the dataset into training and testing sets
    return train_test_split(df, test_size=test_size, random_state=random_state)


def separate_features_target(df, target_column):
    # Separate features and target variable
    features = df.drop([target_column], axis=1)
    target = df[target_column].copy()
    return features, target


if __name__ == "__main__":
    # Load the dataset
    data_frame = pd.read_csv("winequality-white.csv", sep=";")

    # Display the first 100 rows of the dataset
    # print(data_frame.head(100))

    # Split the dataset into training and testing sets
    train_set, test_set = split_data(data_frame)

    # Separate features and target variable in the training set
    X_train, y_train = separate_features_target(train_set, target_column="quality")

    # Separate features and target variable in the training set
    X_test, y_test = separate_features_target(test_set, target_column="quality")

    # Define the hyperparameter optimization function
    def hyperopt_rf(n_estimators, max_depth, min_samples_split, max_features):
        estimator = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),
            random_state=42
        )
        cval = cross_val_score(estimator, X_train, y_train, scoring='accuracy', cv=5)
        return cval.mean()


    def hyperopt_svm(C, gamma):
        estimator = SVC(C=C, gamma=gamma, random_state=42)
        cval = cross_val_score(estimator, X_train, y_train, scoring='accuracy', cv=5)
        return cval.mean()


    def hyperopt_gb(n_estimators, learning_rate, max_depth):
        estimator = GradientBoostingClassifier(
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            random_state=42
        )
        cval = cross_val_score(estimator, X_train, y_train, scoring='accuracy', cv=5)
        return cval.mean()


    # Use Bayesian Optimization to set a range for the hyperparameters
    rf_bo = BayesianOptimization(hyperopt_rf, {
        'n_estimators': (10, 250),
        'max_depth': (1, 50),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
    })

    svm_bo = BayesianOptimization(hyperopt_svm, {
        'C': (0.1, 100),
        'gamma': (0.001, 10.0)
    })
    #
    gb_bo = BayesianOptimization(hyperopt_gb, {
        'n_estimators': (10, 100),
        'learning_rate': (0.01, 0.3),
        'max_depth': (5, 40)
    })

    # Perform Bayesian optimization for each algorithm
    rf_bo.maximize(init_points=5, n_iter=10)
    svm_bo.maximize(init_points=5, n_iter=10)
    gb_bo.maximize(init_points=5, n_iter=10)

    # Get the best hyperparameters and their corresponding scores
    best_rf_params = rf_bo.max['params']
    best_rf_score = rf_bo.max['target']

    best_svm_params = svm_bo.max['params']
    best_svm_score = svm_bo.max['target']

    best_gb_params = gb_bo.max['params']
    best_gb_score = gb_bo.max['target']

    # Train models with the best hyperparameters on the entire training dataset
    best_rf_model = RandomForestClassifier(
        n_estimators=int(best_rf_params['n_estimators']),
        max_depth=int(best_rf_params['max_depth']),
        min_samples_split=int(best_rf_params['min_samples_split']),
        max_features=min(best_rf_params['max_features'], 0.999),
        random_state=42
    )
    best_svm_model = SVC(
        C=best_svm_params['C'],
        gamma=best_svm_params['gamma'],
        random_state=42
    )
    best_gb_model = GradientBoostingClassifier(
        n_estimators=int(best_gb_params['n_estimators']),
        learning_rate=best_gb_params['learning_rate'],
        max_depth=int(best_gb_params['max_depth']),
        random_state=42
    )

    best_rf_model.fit(X_train, y_train)
    best_svm_model.fit(X_train, y_train)
    best_gb_model.fit(X_train, y_train)

    # Evaluate models on the test dataset
    rf_test_score = best_rf_model.score(X_test, y_test)
    svm_test_score = best_svm_model.score(X_test, y_test)
    gb_test_score = best_gb_model.score(X_test, y_test)

    print("Best Random Forest hyperparameters:", best_rf_params)
    print("Best Random Forest score:", best_rf_score)
    print("Random Forest Test Score:", rf_test_score)

    print("Best SVM hyperparameters:", best_svm_params)
    print("Best SVM score:", best_svm_score)
    print("SVM Test Score:", svm_test_score)

    print("Best Gradient Boosting hyperparameters:", best_gb_params)
    print("Best Gradient Boosting score:", best_gb_score)
    print("Gradient Boosting Test Score:", gb_test_score)


# Create a bar plot to visualize the accuracy results
    model_names = ["Random Forest", "SVM", "Gradient Boosting"]
    model_best_scores = [best_rf_score, best_svm_score, best_gb_score]
    model_test_scores = [rf_test_score, svm_test_score, gb_test_score]

    bar_width = 0.35
    index = np.arange(len(model_names))

    plt.figure(figsize=(12, 6))
    bar1 = plt.bar(index, model_best_scores, bar_width, label='Best scores')
    bar2 = plt.bar(index + bar_width, model_test_scores, bar_width, label='Test Scores)')

    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Best scores and Test scores of Different Models')
    plt.xticks(index + bar_width / 2, model_names, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()