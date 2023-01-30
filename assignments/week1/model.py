import numpy as np


class LinearRegression:
    """
    Implementation of Linear Regression without using sklearn
    """

    w: np.ndarray
    b: float

    def __init__(self) -> None:

        """Defining the constructor for Linear Regression"""
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        """
        Fit the model to X and y

        Args:
          X (np.ndarray): The input data
          y (np.ndarray): Target values

        Return: None
        """

        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        self.w = np.dot(
            np.linalg.inv(np.dot(X_centered.T, X_centered)),
            np.dot(X_centered.T, y - y_mean),
        )
        self.b = y_mean - np.dot(self.w, X_mean)

    def predict(self, X: np.ndarray) -> np.ndarray:

        """
        Getting predictions for input data X

        Args:
          X (np.ndarray): The input data

        Return:
          np.ndarray: Array of predicted values
        """
        return np.dot(X, self.w) + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def __init__(self):
        super().__init__()

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit the model to X and y

        Args:
          X (np.ndarray): The input data
          y (np.ndarray): Target values
          lr (float): Learning rate. Default value = 0.01
          epochs (int): Number of epochs. Default value = 1000

        Return: None
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(epochs):
            y_pred = self.predict(X)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= lr * dw
            self.b -= lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return np.dot(X, self.w) + self.b
