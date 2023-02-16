import numpy as np


class LinearRegression:
    """
    Implementation of multiple linear regression.
    """

    def __init__(self):
        self.w: np.ndarray = None  # Weights
        self.b: float = None  # Bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model using the closed form solution.
        Args:
            X (np.ndarray): Input array of shape (n, p), where n is the no. of
            samples and p is the no. of features.
            y (np.ndarray): Output array of shape (n,).
        """
        X_: np.ndarray = np.hstack((np.ones((len(X), 1)), X))  # Add bias column
        w_: np.ndarray = np.linalg.inv(X_.T @ X_) @ X_.T @ y.reshape(-1, 1)
        self.w = w_[1:]
        self.b = w_[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the output for input `X`.
        Arguments:
            X (np.ndarray): Input array of shape (n, p), where n is the no. of
            samples and p is the no. of features.
        Returns:
            np.ndarray: Predicted output.
        """
        return ((X @ self.w) + self.b).reshape(-1)


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
        Fits the model using gradient descent.
        Args:
            X (np.ndarray): Input array of shape (n, p), where n is the no. of
            samples and p is the no. of features.
            y (np.ndarray): Output array of shape (n,).
            lr (float, optional): Learning rate. Defaults to 0.01.
            epochs (int, optional): No. of epochs. Defaults to 1000.
        """
        n: int = X.shape[0]  # No. of samples
        p: int = X.shape[1]  # No. of features
        X_: np.ndarray = np.hstack((np.ones((n, 1)), X))  # Add bias column
        y_: np.ndarray = y.reshape(-1, 1)
        # Randomly initialize weights using a standard normal distribution
        w_: np.ndarray = np.random.randn(p + 1, 1)

        # Fit model using gradient descent
        for _ in range(epochs):
            y_pred: np.ndarray = X_ @ w_
            grad: np.ndarray = (2 / n) * (X_.T @ (y_pred - y_))
            w_ -= lr * grad

        self.w = w_[1:]
        self.b = w_[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the output for input `X`.
        Arguments:
            X (np.ndarray): Input array of shape (n, p), where n is the no. of
            samples and p is the no. of features.
        Returns:
            np.ndarray: Predicted output.
        """
        return super().predict(X)
