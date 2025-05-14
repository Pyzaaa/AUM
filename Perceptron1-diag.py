import numpy as np
import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns
import plotly

class Perceptron:
    # learning_rate to hiperparametr, który kontroluje jak szybko zmieniają się wartości wag, wartość z przedziału (0,1).
    # n - maksymalna liczba epok
    def __init__(self, learning_rate=0.01,  n=1000, tol=1e-4):
        self.lr = learning_rate
        self.n = n
        self.tol = tol  # minimalna zmiana wag, aby kontynuować trening

        self.weights = None  # wagi (będą zainicjalizowane w fit)
        self.bias = None  # bias (próg)
        self.errors_ = []       # liczba błędów w każdej epoce

    def fit(self, X, y):

        # X - dane treningowe liczba próbek x liczba cech
        # y - etykiety 0/1

        n_samples, n_features = X.shape

        # Zainicjalizuj wagi jako zero lub małe wartości losowe.
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Iteracja przez dane treningowe
        for epoch in range(self.n):
            weight_updates = np.zeros_like(self.weights)  # inicjalizacja zmiany wag na 0
            bias_update = 0
            errors = 0
            for target_output_id, x_i in enumerate(X):
                # oblicz iloczyn skalarny pomiędzy wektorem wag a wektorem cech.
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Za pomocą funkcji skokowej Heaviside’a wyznacz klasę.
                y_predicted = self.unit_step_func(linear_output)
                # Jeżeli predykcja dla danej próbki jest zgodna z prawdziwą klasą, przejdź do kolejnej próbki.
                # Jeżeli nie, zaktualizuj wagi
                # w_i(t+1) = w_i(t) + learning_rate * (target_output - predicted_output) * x_{j,i}
                update = self.lr * (y[target_output_id] - y_predicted)

                weight_updates += update * x_i  # sumowanie zmiany wag
                bias_update += update  # sumowanie zmiany bias
                if update != 0:
                    errors += 1


                # Aktualizowanie wag i bias
                self.weights += weight_updates
                self.bias += bias_update
                #if (y[target_output_id] - y_predicted):
                #    print(f"weight update: {weight_updates}; bias update: {bias_update}")

            self.errors_.append(errors)
            print(f"Epoka {epoch + 1:>3}: błędy = {errors}, bias = {self.bias:.4f}, wagi = {self.weights}")

            if errors == 0:
                print(f"Uczenie zakończone wcześniej po {epoch + 1} epokach.")

            # Sprawdzenie, czy zmiana wag jest poniżej progu (tol)
            if np.linalg.norm(weight_updates) < self.tol:
                print(f"Zatrzymano po {epoch+1} epokach, ponieważ zmiana wag jest minimalna.")
                break



    def predict(self, X):
        # oblicz iloczyn skalarny pomiędzy wektorem wag a wektorem cech.
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.unit_step_func(linear_output)
        return y_predicted

    def unit_step_func(self, x):
        # Funkcja skokowa (Heaviside’a).
        # Zwraca 1, jeśli x >= 0, w przeciwnym razie 0.
        return np.where(x >= 0, 1, 0)




def wersja_podstawowa():
    pass

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
    X.hist(bins=30, figsize=(12, 8), layout=(2, 2), edgecolor='black')
    plt.suptitle("Histogramy cech", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Wykres rozkładu rzędów wielkości (log10 wartości bezwzględnych)
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


from sklearn.manifold import TSNE
def podstawowy_perceptron(X, y):
    # Konwersja do numpy array
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()  # flatten do (n,)

    tsne = TSNE(n_components=2)
    X_embedded = tsne.fit_transform(X_np)
    # Podział danych
    X_train, X_test, y_train, y_test = train_test_split(X_embedded, y_np, test_size=0.2, random_state=1)


    # Tworzymy perceptron i trenujemy go na danych 2D
    perceptron = Perceptron(learning_rate=0.01,  n=1000, tol=1e-4)
    perceptron.fit(X_train, y_train)  # Trening na danych po PCA
    y_pred = perceptron.predict(X_test)

    # Ocena
    print("Dokładność:", accuracy_score(y_test, y_pred))
    print("Precyzja:", precision_score(y_test, y_pred))
    print("Czułość (Recall):", recall_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred))

    # Wizualizacja granic decyzyjnych
    plot_decision_boundary(X_train, y_train, perceptron)


def plot_decision_boundary(X, y, model):
    # Tworzenie siatki punktów
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Przewidywanie dla wszystkich punktów w siatce
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Rysowanie granic decyzyjnych
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)

    # Rysowanie punktów
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title("Granice decyzyjne perceptronu (po PCA)")
    plt.xlabel('Pierwsza składowa PCA')
    plt.ylabel('Druga składowa PCA')
    plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def analiza_learning_rate(X, y, lr_values=None, n_epochs=30):
    if lr_values is None:
        lr_values = np.logspace(-4, 0, 20)  # np. od 0.0001 do 1
        lr_values = [0.01, 0.1, 1, 0.001]

    scores = {
        "learning_rate": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=1)

    for lr in lr_values:
        clf = Perceptron(learning_rate=lr, n=n_epochs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        scores["learning_rate"].append(lr)
        scores["accuracy"].append(accuracy_score(y_test, y_pred))
        scores["precision"].append(precision_score(y_test, y_pred))
        scores["recall"].append(recall_score(y_test, y_pred))
        scores["f1"].append(f1_score(y_test, y_pred))

    # Wykres
    plt.figure(figsize=(10, 6))
    for metric in ["accuracy", "precision", "recall", "f1"]:
        plt.plot(scores["learning_rate"], scores[metric], label=metric)

    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Wartość metryki")
    plt.title("Wpływ learning_rate na skuteczność Perceptronu")
    plt.legend()
    plt.grid(True)
    plt.show()

    return pd.DataFrame(scores)


def visualize_decision_boundaries(X, y):
    """
    Funkcja wizualizująca granice decyzyjne dla klasyfikatora perceptronu.

    Parametry:
    X : ndarray
        Cechy wejściowe, powinny mieć dwie kolumny (dwa wymiary).
    y : ndarray
        Etykiety klas.
    """
    # Skalowanie danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tworzymy model perceptronu
    model = Perceptron(learning_rate=0.01, n=100)
    model.fit(X_scaled, y)

    # Tworzymy siatkę punktów na wykresie (granic decyzyjnych)
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Przewidujemy klasy dla siatki punktów
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Rysowanie wykresu
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k', marker='o')
    plt.title("Granica decyzyjna Perceptronu")
    plt.xlabel("Cecha 1")
    plt.ylabel("Cecha 2")
    plt.show()




if __name__ == "__main__":

    # import the dataset from uci
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    banknote_authentication = fetch_ucirepo(id=267)

    # data (as pandas dataframes)
    X = banknote_authentication.data.features
    y = banknote_authentication.data.targets

    # metadata
    print(banknote_authentication.metadata)

    # variable information
    print(banknote_authentication.variables)

    Xpd = pd.DataFrame(banknote_authentication.data.features, columns=banknote_authentication.feature_names)

    #analiza_danych(X)

    from sklearn.preprocessing import StandardScaler, RobustScaler

    # Dopasowanie i transformacja danych
    # StandardScaler
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    # Robust
    X_robust = pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns)

    # Podgląd
    print(X_scaled.head())
    #analiza_danych(X_scaled)

    # Podgląd
    print(X_robust.head())
    #analiza_danych(X_robust)

    # Dane y
    y_np = y.to_numpy().ravel()  # flatten do (n,)
    '''
    sns.countplot(x=y_np)
    plt.title("Rozkład klas")
    plt.show()
    '''
    print("\nWyniki bez skalowania")
    podstawowy_perceptron(X,y)

    print("\nWyniki ze standardowym skalowaniem")
    #podstawowy_perceptron(X_scaled,y)

    print("\nWyniki z robust skalowaniem")
    #podstawowy_perceptron(X_robust,y)

    print("\nAnaliza learning_rate dla danych ze standardowym skalowaniem:")
    #wyniki_lr = analiza_learning_rate(X, y)
    #print(wyniki_lr)


