from sklearn.datasets import load_wine
import pandas as pd

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


class KNNClassifier:
    def __init__(self, k=3, metric='euclidean'):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass

    def accuracy(y_true, y_pred):
        """Oblicza dokładność klasyfikacji."""
        pass

    def precision(y_true, y_pred, average='macro'):
        """Oblicza precyzję. Obsługuje micro/macro averaging."""
        pass

    def recall(y_true, y_pred, average='macro'):
        """Oblicza recall. Obsługuje micro/macro averaging."""
        pass

    def f1_score(y_true, y_pred, average='macro'):
        """Oblicza F1-score. Obsługuje micro/macro averaging."""
        pass