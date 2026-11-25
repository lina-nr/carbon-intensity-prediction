"""
Predicting the carbon intensity of France's electricity mix.

Naive Bayes, k-nearest neighbors and k-fold cross-validation,
implemented from scratch (NumPy and pandas are used only for data handling).
"""

import time

import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------

def entrainer_bayes(train, cible, caracteristiques):
    """Train: compute class priors and conditional probabilities."""
    # Prior probability of each class
    prob_classes = train[cible].value_counts() / len(train)

    # Conditional probabilities P(value | class) for each feature
    prob_cond = {}
    for caract in caracteristiques:
        for c in train[cible].unique():
            sous_ensemble = train[train[cible] == c]
            prob_cond[(c, caract)] = (
                sous_ensemble[caract].value_counts() / len(sous_ensemble)
            )

    return prob_classes, prob_cond


def predire_bayes(donnees, prob_classes, prob_cond, caracteristiques):
    """Predict: for each row, pick the class with the highest score."""
    predictions = []
    for _, ligne in donnees.iterrows():
        prob_max = -1
        classe_predite = None

        for c in prob_classes.index:
            prob = prob_classes[c]
            # Multiply by the conditional probability of each feature
            for caract in caracteristiques:
                prob *= prob_cond[(c, caract)].get(ligne[caract], 0)

            if prob > prob_max:
                prob_max = prob
                classe_predite = c

        predictions.append(classe_predite)

    donnees["Prediction"] = predictions
    return donnees


# ---------------------------------------------------------------
# k-nearest neighbors
# ---------------------------------------------------------------

def distance_euclidienne_vect(point1, point2):
    """Euclidean distance between one point and every row of point1."""
    return np.sqrt(np.sum((point1 - point2) ** 2, axis=1))


def knn_prediction(train, prev, colonnes_explicatives, colonne_a_predire, k):
    """Predict the majority class among the k nearest training points."""
    predictions = []

    train_data = train[colonnes_explicatives].to_numpy()
    prev_data = prev[colonnes_explicatives].to_numpy()

    for i in range(prev_data.shape[0]):
        distances = distance_euclidienne_vect(train_data, prev_data[i])
        # The k closest neighbors
        k_voisins = train.loc[np.argsort(distances)[:k]][colonne_a_predire]
        # Most frequent class among them
        predictions.append(k_voisins.mode()[0])

    prev["Prediction"] = predictions
    return prev


# ---------------------------------------------------------------
# k-fold cross-validation
# ---------------------------------------------------------------

def k_fold_cross_validation(data, colonnes_explicatives, colonne_a_predire, k, n_splits=3):
    """Estimate k-NN accuracy with n-fold cross-validation."""
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    split_size = len(data) // n_splits
    accuracy_scores = []
    start_time = time.time()

    for i in range(n_splits):
        test_start = i * split_size
        test_end = min(test_start + split_size, len(data))

        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

        train_data = data.iloc[train_indices].reset_index(drop=True)
        test_data = data.iloc[test_indices].reset_index(drop=True)

        test_predictions = knn_prediction(
            train_data, test_data.copy(), colonnes_explicatives, colonne_a_predire, k
        )

        accuracy = np.mean(test_predictions["Prediction"] == test_data[colonne_a_predire])
        accuracy_scores.append(accuracy)

    execution_time = time.time() - start_time

    print("Accuracy per fold:", accuracy_scores)
    print("Mean accuracy:", np.mean(accuracy_scores))
    print(f"Execution time: {execution_time:.2f} seconds")

    return accuracy_scores


# ---------------------------------------------------------------
# Example run
# ---------------------------------------------------------------

if __name__ == "__main__":
    # Adjust these file names to match the files downloaded from Kaggle
    train = pd.read_csv("train.csv")
    prev = pd.read_csv("test.csv")

    cible = "MixProdElec"

    # Naive Bayes on temperature and wind speed
    caracteristiques = ["Temperature", "WindSpeed"]
    prob_classes, prob_cond = entrainer_bayes(train, cible, caracteristiques)
    prev_bayes = predire_bayes(prev.copy(), prob_classes, prob_cond, caracteristiques)
    print(prev_bayes["Prediction"].value_counts())

    # k-NN cross-validation on a subset (k-NN is slow on ~100,000 rows)
    train_subset = train[60000:].reset_index(drop=True)
    k_fold_cross_validation(train_subset, ["Temperature", "WindSpeed"], cible, k=5, n_splits=3)
