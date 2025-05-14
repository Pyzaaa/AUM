import numpy as np
import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns
import plotly

class Perceptron:
    # learning_rate to hiperparametr, który kontroluje jak szybko zmieniają się wartości wag, wartość z przedziału (0,1).
    # n - maksymalna liczba epok
    def __init__(self, learning_rate=0.001,  n=1000, tol=1e-4):
        self.lr = learning_rate
        self.n = n
        self.tol = tol  # minimalna zmiana wag, aby kontynuować trening

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

            # Sprawdzenie, czy zmiana wag jest poniżej progu (tol)
            if np.linalg.norm(weight_updates) < self.tol:
                print(f"Zatrzymano po {epoch+1} epokach, ponieważ zmiana wag jest minimalna.")
                break
            #print(f"Epoka {epoch + 1:>3}: błędy = {errors}, bias = {self.bias:.4f}, wagi = {self.weights}")

        print(f"Epoka {epoch + 1:>3}: błędy = {errors}, bias = {self.bias:.4f}, wagi = {self.weights}")
        self.final_epoch = epoch



    def predict(self, X):
        # oblicz iloczyn skalarny pomiędzy wektorem wag a wektorem cech.
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.unit_step_func(linear_output)
        return y_predicted

    def unit_step_func(self, x):
        # Funkcja skokowa (Heaviside’a).
        # Zwraca 1, jeśli x >= 0, w przeciwnym razie 0.
        return np.where(x >= 0, 1, 0)

    def get_epoch_count(self):
        return self.final_epoch+1

    def get_weights(self):
        return (self.bias, self.weights)


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


def podstawowy_perceptron(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Konwersja do numpy array
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()  # flatten do (n,)


    # Podział danych
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=1)

    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)

    # Ocena
    print("Dokładność:", accuracy_score(y_test, y_pred))
    print("Precyzja:", precision_score(y_test, y_pred))
    print("Czułość (Recall):", recall_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def analiza_learning_rate(X, y, lr_values=None, n_epochs=500, test = 0):
    if lr_values is None:
        lr_values = np.logspace(-4, 0, 20)  # np. od 0.0001 do 1
        #lr_values = [0.0001, 0.000162, 0.001, 0.1, 0.5, 1]

    tol_1 = 1e-3
    tol_2 = 1e-2
    scores = {
        "learning_rate": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "epoch_count": []
    }

    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()

    if test:
        from sklearn.datasets import make_classification

        # Tworzymy przykładowe dane
        X_np, y_np = make_classification(n_samples=130, n_features=2, n_informative=2, n_redundant=0, random_state=42)


    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

    # Wyniki dla dwóch wartości tol
    def init_result_dict():
        return {"learning_rate": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "epoch_count": []}

    results = {"tol_1": init_result_dict(), "tol_2": init_result_dict()}

    for lr in lr_values:
        for tol_key, tol in zip(["tol_1", "tol_2"], [tol_1, tol_2]):
            model = Perceptron(learning_rate=lr, n=n_epochs, tol=tol)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[tol_key]["learning_rate"].append(lr)
            results[tol_key]["accuracy"].append(accuracy_score(y_test, y_pred))
            results[tol_key]["precision"].append(precision_score(y_test, y_pred, zero_division=0))
            results[tol_key]["recall"].append(recall_score(y_test, y_pred, zero_division=0))
            results[tol_key]["f1"].append(f1_score(y_test, y_pred, zero_division=0))
            results[tol_key]["epoch_count"].append(model.get_epoch_count())

    # Wykres metryk (accuracy, precision, recall, f1)
    plt.figure(figsize=(12, 7))
    colors = {
        "accuracy": "blue",
        "precision": "green",
        "recall": "red",
        "f1": "black"
    }

    for metric in ["accuracy", "precision", "recall", "f1"]:
        plt.plot(results["tol_1"]["learning_rate"], results["tol_1"][metric],
                 label=f"{metric} (tol={tol_1})", color=colors[metric], linestyle='-')
        plt.plot(results["tol_2"]["learning_rate"], results["tol_2"][metric],
                 label=f"{metric} (tol={tol_2})", color=colors[metric], linestyle='--')

    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Wartość metryki")
    plt.title("Porównanie metryk dla dwóch wartości tolerancji")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Wykres liczby epok
    plt.figure(figsize=(10, 6))
    plt.plot(results["tol_1"]["learning_rate"], results["tol_1"]["epoch_count"],
             label=f"Epoki (tol={tol_1})", color="purple", linestyle='-', marker='o')
    plt.plot(results["tol_2"]["learning_rate"], results["tol_2"]["epoch_count"],
             label=f"Epoki (tol={tol_2})", color="orange", linestyle='--', marker='x')

    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Liczba epok")
    plt.title("Liczba epok vs Learning Rate (dla dwóch tol)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Zwrot DataFrame'ów
    df_tol1 = pd.DataFrame(results["tol_1"])
    df_tol2 = pd.DataFrame(results["tol_2"])
    df_tol1["tol"] = tol_1
    df_tol2["tol"] = tol_2

    return pd.concat([df_tol1, df_tol2], ignore_index=True)


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
    plt.title("Granice decyzyjne perceptronu (po TSNE)")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Alternatywna funkcja rysująca granicę decyzyjną jako realną funkcję liniową na podstawie danych wag i biasu
def plot_decision_boundary_function(X, y, bias, weights):

    # Tworzymy zakres wartości x1 do narysowania prostej
    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    x_vals = np.linspace(x_min, x_max, 200)

    # Obliczamy odpowiadające wartości x2 z równania granicy
    # w1*x1 + w2*x2 + b = 0  =>  x2 = (-w1*x1 - b) / w2
    w1, w2 = weights
    y_vals = (-w1 * x_vals - bias) / w2

    # Rysujemy dane treningowe
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Klasa 0', marker='o', c='blue')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Klasa 1', marker='x', c='green')

    # Rysujemy granicę decyzyjną
    plt.plot(x_vals, y_vals, 'r--', label='Granica decyzyjna')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Granica decyzyjna perceptronu')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualisation(X,y):
    from sklearn.manifold import TSNE


    # Konwersja do numpy array
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()  # flatten do (n,)

    tsne = TSNE(n_components=2)
    X_embedded = tsne.fit_transform(X_np)
    # Podział danych
    X_train, X_test, y_train, y_test = train_test_split(X_embedded, y_np, test_size=0.2, random_state=1)

    # Tworzymy perceptron i trenujemy go na danych 2D
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)  # Trening na danych po TSNE

    # Wizualizacja granic decyzyjnych
    plot_decision_boundary(X_train, y_train, perceptron)
    bias, weights = perceptron.get_weights()
    #plot_decision_boundary_function(X_train, y_train, bias, weights)

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


    from sklearn.preprocessing import StandardScaler, RobustScaler

    # Dopasowanie i transformacja danych
    # StandardScaler
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    # Robust
    X_robust = pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns)


    print(X.head())
    analiza_danych(X)

    # Podgląd
    print(X_scaled.head())
    analiza_danych(X_scaled)

    # Podgląd
    print(X_robust.head())
    analiza_danych(X_robust)

    # Dane y
    y_np = y.to_numpy().ravel()  # flatten do (n,)
    
    sns.countplot(x=y_np)
    plt.title("Rozkład klas")
    plt.show()

    print("\nWyniki bez skalowania")
    podstawowy_perceptron(X,y)

    print("\nWyniki ze standardowym skalowaniem")
    podstawowy_perceptron(X_scaled,y)

    print("\nWyniki z robust skalowaniem")
    podstawowy_perceptron(X_robust,y)

    
    print("\nAnaliza learning_rate dla danych ze standardowym skalowaniem:")
    wyniki_lr = analiza_learning_rate(X_scaled, y)

    # Ustawienia Pandas
    pd.set_option('display.max_columns', None)
    print(wyniki_lr)



    print("Wizualizacja granic decyzyjnych")
    visualisation(X_scaled,y)


