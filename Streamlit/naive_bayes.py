import numpy as np
import pandas as pd
import dill as pickle

class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / len(y)  # P(y)
            self.class_means[cls] = np.mean(X_cls, axis=0)  # Mean per feature
            self.class_variances[cls] = np.maximum(np.var(X_cls, axis=0), 1e-2)  # Variance per feature

    def gaussian_pdf(self, x, mean, var):
        """Compute Gaussian probability density function."""
        var = np.where(var < 1e-2, 1e-2, var) # Avoid division by zero
        eps = 1e-9
        prob = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))
        return np.clip(prob, eps, None)

    def predict(self, X):
        predictions = []
        for x in X:
            class_probs = {}
            for cls in self.classes:
                prior = np.log(self.class_priors[cls])  # Log prior for numerical stability
                likelihoods = np.sum(np.log(self.gaussian_pdf(x, self.class_means[cls], self.class_variances[cls])))
                class_probs[cls] = prior + likelihoods  # Log Posterior = Log Prior + Log Likelihood

            predictions.append(max(class_probs, key=class_probs.get))  # Choose max posterior
        return np.array(predictions)

    def save(self, filepath):
        """Save the model to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """Load the model from a file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

NaiveBayesClassifier.__module__ = '__main__'  # This helps with pickle serialization