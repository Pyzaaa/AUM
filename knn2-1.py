from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

class KNNClassifier:
    def __init__(self, k=3, metric=2):
        self.k = k
        self.metric = metric
        pass

    import numpy as np

    def calculate_distance(self, p1, p2):
        # Set p based on the chosen metric
        p = self.metric

        # Minkowski distance with the determined p value
        dist = np.sum(np.abs(p1 - p2) ** p) ** (1 / p)
        return dist

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self  # Zwracamy obiekt dla zgodności z `sklearn`

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
    """Oblicza dokładność klasyfikacji."""
    correct = sum(y_true == y_pred for y_true, y_pred in zip(y_true, y_pred))
    return correct / len(y_true)


def precision(y_true, y_pred, average='macro'):
    """Oblicza precyzję. Obsługuje micro/macro averaging."""
    classes = np.unique(y_true)
    tp = np.zeros(len(classes))
    fp = np.zeros(len(classes))

    for i, cls in enumerate(classes):
        tp[i] = np.sum((y_pred == cls) & (y_true == cls))
        fp[i] = np.sum((y_pred == cls) & (y_true != cls))

    precision_per_class = tp / (tp + fp)

    if average == 'macro':
        return np.mean(precision_per_class)
    elif average == 'micro':
        return np.sum(tp) / (np.sum(tp) + np.sum(fp))


def recall(y_true, y_pred, average='macro'):
    """Oblicza recall. Obsługuje micro/macro averaging."""
    classes = np.unique(y_true)
    tp = np.zeros(len(classes))
    fn = np.zeros(len(classes))

    for i, cls in enumerate(classes):
        tp[i] = np.sum((y_pred == cls) & (y_true == cls))
        fn[i] = np.sum((y_pred != cls) & (y_true == cls))

    recall_per_class = tp / (tp + fn)

    if average == 'macro':
        return np.mean(recall_per_class)
    elif average == 'micro':
        return np.sum(tp) / (np.sum(tp) + np.sum(fn))


def f1_score(y_true, y_pred, average='macro'):
    """Oblicza F1-score. Obsługuje micro/macro averaging."""
    prec = precision(y_true, y_pred, average)
    rec = recall(y_true, y_pred, average)

    return 2 * (prec * rec) / (prec + rec)

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
def CrossValidate(k_values, p_values):

    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    # Przechowywanie wyników
    results = np.zeros((len(k_values), len(p_values)))  # Macierz wyników
    error_results = np.zeros((len(k_values), len(p_values)))  # Macierz wyników

    best_model = None
    best_accuracy = 0  # Przechowuje najlepszą dotychczasową dokładność

    for i, k in enumerate(k_values):
        for j, p in enumerate(p_values):
            accuracies = []
            errors = []

            all_y_true = []
            all_y_pred = []

            for train_index, test_index in kf.split(X):
                # Podział na zbiór treningowy i testowy
                X_train_fold = X.iloc[train_index].to_numpy()
                X_test_fold = X.iloc[test_index].to_numpy()
                y_train_fold = y.iloc[train_index].to_numpy()
                y_test_fold = y.iloc[test_index].to_numpy()

                # Tworzenie modelu
                knn = KNNClassifier(k=k, metric=p)
                knn.fit(X_train_fold, y_train_fold)
                y_pred_fold = knn.predict(X_test_fold)

                # Zapisujemy prawdziwe i przewidywane wartości do macierzy pomyłek
                all_y_true.extend(y_test_fold)
                all_y_pred.extend(y_pred_fold)

                # Obliczanie dokładności
                acc = accuracy(y_test_fold, y_pred_fold)
                accuracies.append(acc)

                # Błąd
                error = 1 - acc
                errors.append(error)

            # Średnia dokładność dla danej kombinacji k i p
            mean_accuracy = np.mean(accuracies)
            results[i, j] = mean_accuracy

            mean_error = np.mean(errors)
            error_results[i, j] = mean_error
            print(f"k={k}, p={p} -> Średni błąd: {mean_error:.4f}")

            # Aktualizacja najlepszego modelu
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_model = (k, p, all_y_true, all_y_pred)

    # 🔹 Wykres 1: Heatmapa (Macierz wyników k x p)
    plt.figure(figsize=(8, 6))
    sns.heatmap(error_results, annot=True, xticklabels=p_values, yticklabels=k_values, cmap="coolwarm", fmt=".4f")
    plt.xlabel("Wartość p (metryka Minkowskiego)")
    plt.ylabel("Liczba sąsiadów k")
    plt.title("Błąd dla różnych wartości k i p")
    plt.show()

    # 🔹 Wykres 2: Wykresy liniowe dla różnych wartości p
    plt.figure(figsize=(8, 6))
    for j, p in enumerate(p_values):
        plt.plot(k_values, error_results[:, j], marker='o', label=f"p={p}")

    plt.xlabel("Liczba sąsiadów k")
    plt.ylabel("Średni błąd")
    plt.title("Porównanie błędu dla różnych wartości k i p")
    plt.legend()
    plt.grid()
    plt.show()

    # 🔹 Wykres 3: Macierz pomyłek dla najlepszego modelu
    best_k, best_p, y_true_best, y_pred_best = best_model
    cm = confusion_matrix(y_true_best, y_pred_best)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
    plt.xlabel("Przewidywane")
    plt.ylabel("Prawdziwe")
    plt.title(f"Macierz pomyłek najlepszego modelu: (k={best_k}, p={best_p})")
    plt.show()

    return best_model


from sklearn.manifold import TSNE

# 🔹 Funkcja do wizualizacji t-SNE
def visualize_tsne(X, y, k, p):
    """Redukcja wymiarów do 2D za pomocą t-SNE i wizualizacja klasyfikacji."""
    tsne = TSNE(n_components=2, random_state=1)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.75)
    plt.colorbar(scatter, label="Klasa")
    plt.title(f"t-SNE: Redukcja wymiarów (k={k}, p={p})")
    plt.xlabel("Wymiar 1")
    plt.ylabel("Wymiar 2")
    plt.show()


def TestVariables():
    metrics = [1, 1.5, 2, 3, 4, 5]  # List of metrics
    k_values = [1, 2, 3, 4, 5, 7, 9]  # List of k values to test

    best_accuracy = 0  # Initialize the best accuracy
    best_metric = None  # Initialize the best metric
    best_k = None  # Initialize the best k value

    # Loop through all combinations of metrics and k values
    for metric in metrics:
        for k in k_values:
            knn = KNNClassifier(metric=metric, k=k)  # Set metric and k
            knn.fit(X_train.values, y_train.values)
            y_pred = knn.predict(X_test.values)
            current_accuracy = accuracy(y_test.values, y_pred)

            # Print results for current combination
            print(f"Metric: {knn.metric}, k: {knn.k}")
            print(f"Accuracy: {current_accuracy}")

            # Update best result if the current one is better
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_metric = metric
                best_k = k

    # Output the best combination
    print(f"Best Metric: {best_metric}, Best k: {best_k}, Best Accuracy: {best_accuracy}")



'''from sklearn.model_selection import cross_val_score

knn = KNNClassifier(k=3, metric=2)
scores = cross_val_score(knn, X.values, y.values, cv=5)

print("Dokładności w każdej iteracji:", scores)
print("Średnia dokładność:", scores.mean())
'''




if __name__ == "__main__":
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Podział na zbiór treningowy i testowy (80% trening, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    #print(X_test)

    # readable numpy
    np.set_printoptions(precision=3, suppress=True)

    # Twórz classifier
    knn = KNNClassifier()
    # dane z dataframe muszą mieć Values żeby nie podawać nazw kolumn itp.
    knn.fit(X_train.values, y_train.values)

    #print(X_train.values)
    #print(y_train.values)

    y_pred = knn.predict(X_test.values)

    #print(y_pred)
    #print(y_test.values)

    print(accuracy(y_test.values, y_pred))

    print(precision(y_test.values, y_pred))
    print(recall(y_test.values, y_pred))
    print(f1_score(y_test.values, y_pred))


    # Możliwe wartości k i p
    k_values = [1, 2, 3, 5, 7, 9]  # Liczba sąsiadów
    p_values = [1, 1.5, 2, 3, 4, 5]  # Metryka Minkowskiego (1 = Manhattan, 2 = Euklidesowa itd.)\

    #Cross validate
    best_model = CrossValidate(k_values, p_values)

    # Test variables on single split
    TestVariables()

    # 🔹 Wizualizacja t-SNE dla najlepszego modelu
    if best_model:
        visualize_tsne(X_test_best, y_pred_best, best_k, best_p)