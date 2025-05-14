from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        pass

    import numpy as np

    def calculate_distance(self, p1, p2):
        # Set p based on the chosen metric
        if self.metric == "euclidean":
            p = 2  # Euclidean distance corresponds to p=2
        elif self.metric == "manhattan":
            p = 1  # Manhattan distance corresponds to p=1
        elif self.metric == "minkowski":
            p = 3  # You can set p to any other value for Minkowski distance
        elif self.metric == "1.5":
            p = 1.5  # You can set p to any other value for Minkowski distance
        elif self.metric == "4":
            p = 4 # You can set p to any other value for Minkowski distance
        elif self.metric == "5":
            p = 5  # You can set p to any other value for Minkowski distance
        else:
            p = 2  # Default to Euclidean if no valid metric is specified

        # Minkowski distance with the determined p value
        dist = np.sum(np.abs(p1 - p2) ** p) ** (1 / p)
        return dist

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        pass

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            try:
                distances = [self.calculate_distance(x, trained) for trained in self.X_train]
                neighbors = np.argsort(distances)[:self.k]
                k_nearest_neighbors = [self.y_train[i] for i in neighbors]
                most_common = Counter(k_nearest_neighbors).most_common(1)[0][0]
                predictions.append(most_common)
            except:
                pass
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

    precision_per_class = tp / (tp + fp + 1e-9)  # Dodajemy epsilon, aby uniknąć dzielenia przez zero

    if average == 'macro':
        return np.mean(precision_per_class)
    elif average == 'micro':
        return np.sum(tp) / (np.sum(tp) + np.sum(fp) + 1e-9)
    else:
        raise ValueError("average must be 'micro' or 'macro'")


def recall(y_true, y_pred, average='macro'):
    """Oblicza recall. Obsługuje micro/macro averaging."""
    classes = np.unique(y_true)
    tp = np.zeros(len(classes))
    fn = np.zeros(len(classes))

    for i, cls in enumerate(classes):
        tp[i] = np.sum((y_pred == cls) & (y_true == cls))
        fn[i] = np.sum((y_pred != cls) & (y_true == cls))

    recall_per_class = tp / (tp + fn + 1e-9)

    if average == 'macro':
        return np.mean(recall_per_class)
    elif average == 'micro':
        return np.sum(tp) / (np.sum(tp) + np.sum(fn) + 1e-9)
    else:
        raise ValueError("average must be 'micro' or 'macro'")


def f1_score(y_true, y_pred, average='macro'):
    """Oblicza F1-score. Obsługuje micro/macro averaging."""
    prec = precision(y_true, y_pred, average)
    rec = recall(y_true, y_pred, average)

    return 2 * (prec * rec) / (prec + rec + 1e-9)

from sklearn.model_selection import train_test_split

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

from sklearn.model_selection import cross_val_score
'''
scores = cross_val_score(knn, X_train.values, y_train.values, cv=5)

print("Dokładności w każdej iteracji:", scores)
print("Średnia dokładność:", scores.mean())


'''
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Redukcja danych testowych do 2D
X_test_2D = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_test.values)

# Wizualizacja klasyfikacji
plt.scatter(X_test_2D[:, 0], X_test_2D[:, 1], c=y_pred, cmap='coolwarm', alpha=0.7)
plt.colorbar(label="Przewidywana klasa")
plt.xlabel("Wymiar 1")
plt.ylabel("Wymiar 2")
plt.title("Wizualizacja klasyfikacji (t-SNE)")
plt.show()


def TestVariables():
    metrics = ['euclidean', 'minkowski', 'manhattan', '1.5', '4', '5']  # List of metrics
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


# TestVariables()

from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(X_test)

#print(X_embedded)

import matplotlib.pyplot as plt

# Wizualizacja w 2D
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_test, cmap='viridis', alpha=0.7)
plt.colorbar(label="Klasy")

plt.xlabel("Wymiar 1")
plt.ylabel("Wymiar 2")
plt.title("t-SNE - Wizualizacja danych w 2D")
plt.show()
