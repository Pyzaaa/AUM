import numpy as np
import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns
import plotly

class Perceptron:
    # learning_rate to hiperparametr, który kontroluje jak szybko zmieniają się wartości wag, wartość z przedziału (0,1).
    # n - maksymalna liczba epok
    def __init__(self, learning_rate=0.001,  n=1000):
        self.lr = learning_rate
        self.n = n

        self.weights = None  # wagi (będą zainicjalizowane w fit)
        self.bias = None  # bias (próg)
        self.final_epoch = 0

    def fit(self, X, y):

        # X - dane treningowe liczba próbek x liczba cech
        # y - etykiety 0/1

        n_samples, n_features = X.shape

        # Zainicjalizuj wagi jako zero lub małe wartości losowe.
        self.weights = np.zeros(n_features)
        self.bias = 0
        # Iteracja przez dane treningowe
        for epoch in range(self.n):
            errors = 0
            weight_updates = np.zeros_like(self.weights)  # inicjalizacja zmiany wag na 0
            bias_update = 0
            for target_output_id, x_i in enumerate(X):
                # oblicz iloczyn skalarny pomiędzy wektorem wag a wektorem cech.
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Za pomocą funkcji skokowej Heaviside’a wyznacz klasę.
                y_predicted = self.unit_step_func(linear_output)
                # Jeżeli predykcja dla danej próbki jest zgodna z prawdziwą klasą, przejdź do kolejnej próbki.
                # Jeżeli nie, zaktualizuj wagi
                # w_i(t+1) = w_i(t) + learning_rate * (target_output - predicted_output) * x_{j,i}
                update = self.lr * (y[target_output_id] - y_predicted)
                if update != 0:
                    errors += 1

                weight_updates += update * x_i  # sumowanie zmiany wag
                bias_update += update  # sumowanie zmiany bias

            # Aktualizowanie wag i bias
            self.weights += weight_updates
            self.bias += bias_update
            #if (y[target_output_id] - y_predicted):
            #    print(f"weight update: {weight_updates}; bias update: {bias_update}")
            #print(f"Epoka {epoch + 1:>3}: błędy = {errors}, bias = {self.bias:.4f}, wagi = {self.weights}")

        print(f"Epoka {epoch + 1:>3}: błędy = {errors}, bias = {self.bias:.4f}, wagi = {self.weights}")
        self.final_epoch = epoch



    def predict(self, X):
        # oblicz iloczyn skalarny pomiędzy wektorem wag a wektorem cech.
        linear_output = self.predict_proba(X)
        y_predicted = self.unit_step_func(linear_output)
        return y_predicted

    def predict_proba(self, X):
        # oblicz iloczyn skalarny pomiędzy wektorem wag a wektorem cech.
        #print(np.dot(X, self.weights) + self.bias)
        return np.dot(X, self.weights) + self.bias

    def unit_step_func(self, x):
        # Funkcja skokowa (Heaviside’a).
        # Zwraca 1, jeśli x >= 0, w przeciwnym razie 0.
        return np.where(x >= 0, 1, 0)

    def get_epoch_count(self):
        return self.final_epoch+1

    def get_weights(self):
        return (self.bias, self.weights)



class OVRClassifier:
    def __init__(self, learning_rate=0.001, n=1000):
        self.learning_rate = learning_rate
        self.n = n
        self.classifiers = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            print(f'fitting class: {cls}')
            # Przygotuj etykiety binarne: klasa == cls → 1, reszta → 0
            binary_y = (y == cls)
            clf = Perceptron(learning_rate=self.learning_rate, n=self.n)
            clf.fit(X, binary_y)
            self.classifiers[cls] = clf
        return self

    def predict(self, X):
        # Każdy perceptron zwraca wynik surowy (margines decyzyjny)
        scores = np.column_stack([
            self.classifiers[cls].predict_proba(X) for cls in self.classes_
        ])
        # Wybieramy klasę o najwyższym score
        return self.classes_[np.argmax(scores, axis=1)]

from sklearn.svm import SVC

class OVRSVCClassifier:
    def __init__(self, kernel="linear"):
        self.kernel = kernel
        self.classifiers = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            print(f'fitting class: {cls}')
            # Przygotuj etykiety binarne: klasa == cls → 1, reszta → 0
            binary_y = (y == cls)
            print(binary_y)
            clf = SVC(kernel=self.kernel)
            clf.fit(X, binary_y)
            self.classifiers[cls] = clf
        return self

    def predict(self, X):
        # Każdy perceptron zwraca wynik surowy (margines decyzyjny)
        scores = np.column_stack([
            self.classifiers[cls].predict(X) for cls in self.classes_
        ])
        # Wybieramy klasę o najwyższym score
        return self.classes_[np.argmax(scores, axis=1)]




def accuracy(y_true, y_pred):
    """Oblicza dokładność klasyfikacji."""
    correct = sum(y_true == y_pred for y_true, y_pred in zip(y_true, y_pred))
    return correct / len(y_true)


def precision(y_true, y_pred, average='macro'):
    classes = np.unique(y_true)
    tp = np.zeros(len(classes))
    fp = np.zeros(len(classes))

    for i, cls in enumerate(classes):
        tp[i] = np.sum((y_pred == cls) & (y_true == cls))
        fp[i] = np.sum((y_pred == cls) & (y_true != cls))

    precision_per_class = tp / (tp + fp + 1e-10)  # dodany epsilon dla dzielenia przez zero

    if average == 'macro':
        return np.mean(precision_per_class)
    elif average == 'micro':
        return np.sum(tp) / (np.sum(tp) + np.sum(fp) + 1e-10)


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


def analiza_danych(X):
    # Ustawienia Pandas
    pd.set_option('display.max_columns', None)

    print("Podgląd danych X")
    print(X)

    print("\nWariancja cech:")
    print(X.var())

    print("\nZakres wartości (min, max):")
    print(X.describe().loc[['min', 'max']])

    print("\nPodstawowe statystyki opisowe:")
    print(X.describe().T)

    print("\nWartości najbliższe zeru:")
    for col in X.columns:
        closest_to_zero = X[col].iloc[(X[col] - 0).abs().argsort()].iloc[0]
        print(f"{col}: {closest_to_zero}")

    # Histogramy cech
    X.hist(bins=30, figsize=(14, 10), edgecolor='black')
    plt.suptitle("Histogramy cech zbioru win", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Wykres rozkładu rzędów wielkości (log10)
    log_scales = []
    for col in X.columns:
        log_values = np.log10(X[col].abs().replace(0, np.nan).dropna())
        log_scales.extend(np.floor(log_values))

    log_scales = pd.Series(log_scales)
    plt.figure(figsize=(8, 5))
    sns.countplot(x=log_scales.astype(int), palette="viridis")
    plt.title("Rozkład rzędów wielkości (log10 |x|)")
    plt.xlabel("Rząd wielkości (floor(log10 |x|))")
    plt.ylabel("Liczba wartości")
    plt.grid(True)
    plt.show()

def analiza_danych_np(X_np, feature_names):

    X = pd.DataFrame(X_np, columns=feature_names)
    # Ustawienia Pandas
    pd.set_option('display.max_columns', None)

    print("Podgląd danych X")
    print(X)

    print("\nWariancja cech:")
    print(X.var())

    print("\nZakres wartości (min, max):")
    print(X.describe().loc[['min', 'max']])

    print("\nPodstawowe statystyki opisowe:")
    print(X.describe().T)

    print("\nWartości najbliższe zeru:")
    for col in X.columns:
        closest_to_zero = X[col].iloc[(X[col] - 0).abs().argsort()].iloc[0]
        print(f"{col}: {closest_to_zero}")

    # Histogramy cech
    X.hist(bins=30, figsize=(14, 10), edgecolor='black')
    plt.suptitle("Histogramy cech zbioru win", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Wykres rozkładu rzędów wielkości (log10)
    log_scales = []
    for col in X.columns:
        log_values = np.log10(X[col].abs().replace(0, np.nan).dropna())
        log_scales.extend(np.floor(log_values))

    log_scales = pd.Series(log_scales)
    plt.figure(figsize=(8, 5))
    sns.countplot(x=log_scales.astype(int), palette="viridis")
    plt.title("Rozkład rzędów wielkości (log10 |x|)")
    plt.xlabel("Rząd wielkości (floor(log10 |x|))")
    plt.ylabel("Liczba wartości")
    plt.grid(True)
    plt.show()

def rozklad_klas(y):
    # Konwersja y do serii Pandas dla kompatybilności z seaborn
    y_series = pd.Series(y, name="Klasa")

    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_series, palette="pastel")
    plt.title("Rozkład klas w zbiorze")
    plt.xlabel("Klasa")
    plt.ylabel("Liczba próbek")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def wykres_macierzy_pomylek(y_true, y_pred, class_names):
    # Generowanie macierzy pomyłek
    cm = confusion_matrix(y_true, y_pred)

    # Normalizacja macierzy pomyłek (opcjonalnie)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Rysowanie heatmapy
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                cbar=False)

    plt.title("Macierz pomyłek")
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywista klasa")
    plt.tight_layout()
    plt.show()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def rysuj_granice_decyzyjne_pca(ovr_classifier, X, y):
    # Redukcja wymiarów do 2D za pomocą PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Siatka punktów
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # Predykcja na siatce
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Transformujemy z powrotem do oryginalnej przestrzeni cech
    grid_original = pca.inverse_transform(grid)
    Z = ovr_classifier.predict(grid_original)
    Z = Z.reshape(xx.shape)

    # Wykres
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', cmap=plt.cm.Set1)
    plt.legend(*scatter.legend_elements(), title="Klasy")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Granice decyzyjne klasyfikatora OVR (po PCA)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

def plot_ovr_decision_boundaries_tsne(X, y, perplexity=30, random_state=6):

    # 1. Redukcja wymiaru do 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X)

    # 2. Siatka do rysowania granic
    x_min, x_max = X_tsne[:, 0].min() - 5, X_tsne[:, 0].max() + 5
    y_min, y_max = X_tsne[:, 1].min() - 5, X_tsne[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 3. Trenowanie klasyfikatora na przekształconych danych (do celów wizualizacji)
    #OVR_tsne = OVRClassifier(learning_rate=0.001, n=1000)
    OVR_tsne = OVRSVCClassifier()
    OVR_tsne.fit(X_tsne, y)

    # 4. Predykcje na siatce
    Z = OVR_tsne.predict(grid)
    Z = Z.reshape(xx.shape)

    # 5. Wizualizacja
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=40)
    plt.legend(*scatter.legend_elements(), title="Klasy")
    plt.title("Granice decyzyjne (One-vs-Rest) po t-SNE")
    plt.xlabel("Wymiar 1 (t-SNE)")
    plt.ylabel("Wymiar 2 (t-SNE)")
    plt.grid(True)
    plt.show()

def klasyfikator_OVR(X_train, X_test, y_train, y_test):
    # 3. Wytrenuj klasyfikator OVR
    One_VS_Rest = OVRClassifier(learning_rate=0.001, n=1000)
    One_VS_Rest.fit(X_train, y_train)
    y_pred = One_VS_Rest.predict(X_test)

    # 4. Oblicz wskaźniki
    print("Accuracy:", accuracy(y_test, y_pred))
    print("Precision (macro):", precision(y_test, y_pred, average='macro'))
    print("Precision (micro):", precision(y_test, y_pred, average='micro'))
    print("Recall (macro):", recall(y_test, y_pred, average='macro'))
    print("Recall (micro):", recall(y_test, y_pred, average='micro'))
    print("F1-score (macro):", f1_score(y_test, y_pred, average='macro'))
    print("F1-score (micro):", f1_score(y_test, y_pred, average='micro'))

    # Wykres macierzy pomyłek
    class_names = data.target_names  # Klasy w zbiorze win
    wykres_macierzy_pomylek(y_test, y_pred, class_names)

    rysuj_granice_decyzyjne_pca(One_VS_Rest, X_test, y_test)

def klasyfikator_OVRSVC(X_train, X_test, y_train, y_test):
    # 3. Wytrenuj klasyfikator OVR
    One_VS_Rest = OVRSVCClassifier()
    One_VS_Rest.fit(X_train, y_train)
    y_pred = One_VS_Rest.predict(X_test)

    # 4. Oblicz wskaźniki
    print("Accuracy:", accuracy(y_test, y_pred))
    print("Precision (macro):", precision(y_test, y_pred, average='macro'))
    print("Precision (micro):", precision(y_test, y_pred, average='micro'))
    print("Recall (macro):", recall(y_test, y_pred, average='macro'))
    print("Recall (micro):", recall(y_test, y_pred, average='micro'))
    print("F1-score (macro):", f1_score(y_test, y_pred, average='macro'))
    print("F1-score (micro):", f1_score(y_test, y_pred, average='micro'))

    # Wykres macierzy pomyłek
    class_names = data.target_names  # Klasy w zbiorze win
    wykres_macierzy_pomylek(y_test, y_pred, class_names)

    rysuj_granice_decyzyjne_pca(One_VS_Rest, X_test, y_test)

if __name__ == "__main__":
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler


    # 1. Wczytaj dane
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Skalowanie cech
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Analiza danych wejściowych
    #analiza_danych(X)
    #analiza_danych(X_scaled)
    #rozklad_klas(y)

    X = X.to_numpy()
    X_scaled = X_scaled.to_numpy()

    # 2. Podziel dane
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    #X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1, stratify=y)

    #klasyfikator_OVR(X_train, X_test, y_train, y_test)
    #klasyfikator_OVRSVC(X_train, X_test, y_train, y_test)

    plot_ovr_decision_boundaries_tsne(X_scaled,y)