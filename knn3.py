from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold


class KNNClassifier:
    def __init__(self, k=3, metric=2):
        self.k = k
        self.metric = metric

    def calculate_distance(self, p1, p2):
        p = self.metric
        return np.sum(np.abs(p1 - p2) ** p) ** (1 / p)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [self.calculate_distance(x, trained) for trained in self.X_train]
            neighbors = np.argsort(distances)[:self.k]
            k_nearest_neighbors = [self.y_train[i] for i in neighbors]
            most_common = Counter(k_nearest_neighbors).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def CrossValidate(k_values, p_values, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    results = np.zeros((len(k_values), len(p_values)))
    best_model = None
    best_accuracy = 0

    for i, k in enumerate(k_values):
        for j, p in enumerate(p_values):
            accuracies = []
            all_y_true, all_y_pred = [], []

            for train_index, test_index in kf.split(X):
                X_train_fold, X_test_fold = X.iloc[train_index].to_numpy(), X.iloc[test_index].to_numpy()
                y_train_fold, y_test_fold = y.iloc[train_index].to_numpy(), y.iloc[test_index].to_numpy()

                knn = KNNClassifier(k=k, metric=p)
                knn.fit(X_train_fold, y_train_fold)
                y_pred_fold = knn.predict(X_test_fold)

                all_y_true.extend(y_test_fold)
                all_y_pred.extend(y_pred_fold)
                accuracies.append(accuracy(y_test_fold, y_pred_fold))

            mean_accuracy = np.mean(accuracies)
            results[i, j] = mean_accuracy

            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_model = (k, p, all_y_true, all_y_pred)

    return best_model


def visualize_tsne(X, y, k, p):
    tsne = TSNE(n_components=2, random_state=1)
    X_2d = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.75)
    plt.colorbar(scatter, label="Klasa")
    plt.title(f"t-SNE: Redukcja wymiarów (k={k}, p={p})")
    plt.xlabel("Wymiar 1")
    plt.ylabel("Wymiar 2")
    plt.show()


if __name__ == "__main__":
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    knn = KNNClassifier()
    knn.fit(X_train.values, y_train.values)
    y_pred = knn.predict(X_test.values)
    print("Accuracy:", accuracy(y_test.values, y_pred))

    k_values = [1, 2, 3, 5, 7, 9]
    p_values = [1, 1.5, 2, 3, 4, 5]

    best_model = CrossValidate(k_values, p_values, X, y)

    if best_model:
        best_k, best_p, y_true_best, y_pred_best = best_model
        visualize_tsne(X_test.values, y_pred_best, best_k, best_p)
